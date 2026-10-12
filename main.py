from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO


class Config:
    source = "video.mp4"              # input video path
    output = "output.mp4"             # output video path
    model = "yolov8s.pt"              # YOLO weights (COCO or your custom football model, e.g. "best.pt")
    player_classes = ["person"]       # custom model? e.g. ["player", "goalkeeper"]
    conf = 0.3                        # detection confidence
    fit_frames = 10                   # frames used to learn the two team colors
    vote_frames = 15                  # appearances before a player's team is locked
    show = True                       # show live window (press q to stop)


class TeamAssigner:
    def __init__(self, vote_frames: int = 15):
        self.team_colors = {}            # {1: bgr, 2: bgr}  (display colors)
        self.kmeans = None               # 2-cluster classifier (team 1 / team 2)
        self.votes = defaultdict(list)   # track_id -> [1, 1, 2, 1 ...]
        self.player_team = {}            # track_id -> final team
        self.vote_frames = vote_frames

    def get_player_color(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 5:  # height then width
            return None

        # Use only the top half (jersey area, not shorts/legs)
        top_half = crop[: crop.shape[0] // 2, :]
        pixels = top_half.reshape(-1, 3).astype(np.float32)

        # FIX: KMeans(k=2) needs at least 2 distinct colors
        if len(np.unique(pixels, axis=0)) < 2:
            return pixels.mean(axis=0)

        km = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0).fit(pixels)
        labels = km.labels_.reshape(top_half.shape[0], top_half.shape[1])

        # Background cluster = the one that owns the majority of the 4 corners
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - bg_cluster
        return km.cluster_centers_[player_cluster]  # BGR

    # ---------- map a raw color to a clean display color ---------- #
    @staticmethod
    def _display_color(center: np.ndarray) -> np.ndarray:
        b, g, r = center
        if g > b + 25 and g > r + 15:
            return np.array([40, 245, 60], dtype=np.float32)     # Neon Green (BGR)
        if min(b, g, r) > 170:
            return np.array([255, 255, 255], dtype=np.float32)   # White (BGR)
        if r > g + 30 and r > b + 30:
            return np.array([0, 0, 255], dtype=np.float32)       # Red (BGR)
        if b > g + 30 and b > r + 30:
            return np.array([255, 80, 0], dtype=np.float32)      # Blue (BGR)
        return center.astype(np.float32)                         # raw detected color

    # ---------- fit the two team colors ---------- #
    def fit_team_colors(self, frames_and_boxes):
        """frames_and_boxes: list of (frame, [bbox, ...])"""
        colors = []
        for frame, boxes in frames_and_boxes:
            for bbox in boxes:
                c = self.get_player_color(frame, bbox)
                if c is not None:
                    colors.append(c)
        if len(colors) < 4:
            raise RuntimeError("Not enough players detected to fit team colors.")

        colors = np.array(colors)

        # k=3: Team 1, Team 2, and referee / goalkeeper / outliers
        km_initial = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(colors)
        counts = np.bincount(km_initial.labels_, minlength=3)

        # The two largest clusters = the two outfield teams
        top_two = np.argsort(counts)[-2:]
        team1_center = km_initial.cluster_centers_[top_two[0]]
        team2_center = km_initial.cluster_centers_[top_two[1]]

        c1 = self._display_color(team1_center)
        c2 = self._display_color(team2_center)
        # FIX: if both teams map to the same display color, use raw colors instead
        if np.array_equal(c1, c2):
            c1, c2 = team1_center.astype(np.float32), team2_center.astype(np.float32)
        self.team_colors = {1: c1, 2: c2}

        # 2-team classifier initialized with the two discovered team centers
        centers = np.array([team1_center, team2_center])
        self.kmeans = KMeans(n_clusters=2, init=centers, n_init=1, random_state=0)
        self.kmeans.fit(centers)

    # ---------- assign a team to a tracked player ---------- #
    def get_player_team(self, frame, bbox, track_id) -> int | None:
        if track_id in self.player_team:
            return self.player_team[track_id]
        if self.kmeans is None:  # FIX: colors were never fitted
            return None

        color = self.get_player_color(frame, bbox)
        if color is None:
            return self._current_vote(track_id)

        team = int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1
        self.votes[track_id].append(team)

        if len(self.votes[track_id]) >= self.vote_frames:
            self.player_team[track_id] = Counter(self.votes[track_id]).most_common(1)[0][0]
            del self.votes[track_id]
            return self.player_team[track_id]
        return self._current_vote(track_id)

    def _current_vote(self, track_id):
        v = self.votes.get(track_id)
        return Counter(v).most_common(1)[0][0] if v else None


# --------------------------------------------------------------------------- #
#                                 Drawing                                     #
# --------------------------------------------------------------------------- #
def draw_player(frame, bbox, color, track_id):
    x1, y1, x2, y2 = map(int, bbox)
    cx, w = (x1 + x2) // 2, x2 - x1
    color = tuple(int(c) for c in color)

    # 1. Dark outer border for contrast against green pitch
    cv2.ellipse(frame, (cx, y2), (int(w * 0.6), int(0.2 * w)), 0, -45, 235, (0, 0, 0), 4, cv2.LINE_AA)
    # 2. Main team color ellipse
    cv2.ellipse(frame, (cx, y2), (int(w * 0.6), int(0.2 * w)), 0, -45, 235, color, 2, cv2.LINE_AA)

    # ID badge with clean 1px border
    rw, rh = 40, 20
    rx1, ry1 = cx - rw // 2, y2 + 10
    cv2.rectangle(frame, (rx1, ry1), (rx1 + rw, ry1 + rh), color, cv2.FILLED)
    cv2.rectangle(frame, (rx1, ry1), (rx1 + rw, ry1 + rh), (0, 0, 0), 1)

    txt_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(frame, str(track_id), (rx1 + 6, ry1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 2)


def main():
    args = Config

    model = YOLO(args.model)
    player_cls_ids = {i for i, n in model.names.items() if n in args.player_classes}
    if not player_cls_ids:
        raise ValueError(f"None of {args.player_classes} found in model classes: {list(model.names.values())}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened(): 
        raise FileNotFoundError(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    assigner = TeamAssigner(vote_frames=args.vote_frames)
    buffer = []        
    fitted = False

    def detect(frame):
        res = model.track(
            frame, persist=True, conf=args.conf, tracker="bytetrack.yaml",
            classes=sorted(player_cls_ids), verbose=False,
        )[0]
        boxes, ids = [], []
        if res.boxes is not None and res.boxes.id is not None:
            for b, c, t in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.id.cpu().numpy()):
                if int(c) in player_cls_ids:
                    boxes.append(b)
                    ids.append(int(t))
        return boxes, ids

    def render(frame, boxes, ids):
        for bbox, tid in zip(boxes, ids):
            team = assigner.get_player_team(frame, bbox, tid)
            color = assigner.team_colors.get(team, (200, 200, 200))
            draw_player(frame, bbox, color, tid)
        writer.write(frame)
        if args.show:
            cv2.imshow("teams", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    stop = False
    while not stop:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, ids = detect(frame)

        if not fitted:
            buffer.append((frame, boxes, ids))
            if len(buffer) % args.fit_frames == 0:
                try:
                    assigner.fit_team_colors([(f, b) for f, b, _ in buffer])
                except RuntimeError:
                    continue  # not enough players yet -> keep buffering
                fitted = True
                print("Team colors (BGR):", {k: v.astype(int).tolist() for k, v in assigner.team_colors.items()})
                for f, b, i in buffer:
                    if not render(f, b, i):
                        stop = True
                        break
                buffer.clear()
            continue

        if not render(frame, boxes, ids):
            break

    if not fitted and buffer:
        try:
            assigner.fit_team_colors([(f, b) for f, b, _ in buffer])
        except RuntimeError:
            print("Warning: not enough players to learn team colors; drawing in gray.")
        for f, b, i in buffer:
            if not render(f, b, i):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()