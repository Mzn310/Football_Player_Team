from __future__ import annotations

import os
from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))   # paths below are relative to this file


class Config:
    source = "video.mp4"              # input video path
    output = "output_v2.mp4"          # output video path
    model = "best.pt"                 # custom football model: ball / goalkeeper / player / referee
    person_model = "../yolov8s.pt"    # plain COCO model: finds coaches / bench / staff that best.pt never learned (None = off)
    person_imgsz = 1280               # bigger input so small people on the touchline are found
    conf = 0.3                        # detection confidence (best.pt)
    nms_iou = 0.5                     # best.pt boxes overlapping more than this are merged (default 0.7 leaves duplicates)
    person_conf = 0.3                 # detection confidence (COCO person model)
    fit_frames = 10                   # frames used to learn the two team kits
    vote_frames = 15                  # appearances before a track's kit is locked
    gk_min_share = 0.15               # track is a goalkeeper if >= 15% of its detections say "goalkeeper"
    kit_tolerance = 2.0               # kit farther than (typical team spread x this) from every team -> not a team kit
    staff_min_frames = 5              # a person must look like staff this many frames before being drawn
    use_pitch_filter = True           # feet outside the green pitch -> "staff" (and crowd is ignored)
    max_frames = None                 # e.g. 200 for a quick test, None = whole video
    show = True                       # show live window (press q to stop)

    # model class name (lowercase) -> role. Classes not listed here are ignored.
    role_map = {
        "player": "player",
        "football-players": "player",
        "person": "player",           # works with plain COCO too (no goalkeeper/referee then)
        "goalkeeper": "goalkeeper",
        "referee": "referee",
        "coach": "coach",
        "ball": "ball",
    }


ROLE_COLORS = {                       # BGR
    "goalkeeper": (0, 165, 255),      # orange
    "referee":    (0, 255, 255),      # yellow
    "coach":      (255, 255, 0),      # cyan
    "staff":      (255, 0, 255),      # magenta
    "ball":       (0, 0, 255),        # red
}
ROLE_LABELS = {"referee": "REF", "coach": "COACH", "staff": "STAFF"}


def path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(HERE, p)


# --------------------------------------------------------------------------- #
#                               Pitch detection                               #
# --------------------------------------------------------------------------- #
def pitch_polygon(frame: np.ndarray):
    """Green area of the pitch as a convex hull (computed per frame because the camera moves)."""
    s = 0.25
    small = cv2.resize(frame, None, fx=s, fy=s)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    return (hull / s).astype(np.int32)


def is_on_pitch(poly, bbox, margin: float = -8) -> bool:
    if poly is None:
        return True
    x1, y1, x2, y2 = bbox
    d = cv2.pointPolygonTest(poly, (float((x1 + x2) / 2), float(y2)), True)
    return d >= margin                # a few pixels of tolerance near the lines


def overlap(a, b):
    """(IoU, intersection / smaller box) of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    if inter <= 0:
        return 0.0, 0.0
    return inter / (area_a + area_b - inter), inter / min(area_a, area_b)


# --------------------------------------------------------------------------- #
#                                 Kit assigner                                #
# --------------------------------------------------------------------------- #
L_WEIGHT = 0.5                        # lightness counts half: shadows change L much more than the kit's hue


def kit_feature(bgr: np.ndarray) -> np.ndarray:
    """BGR color -> Lab (perceptual: distances match what the eye sees)."""
    px = np.clip(bgr, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    lab[0] *= L_WEIGHT
    return lab


class TeamAssigner:
    """Learns the two team kits and tells which kit a tracked person wears.

    A kit is 1 or 2 (team kits) or 0 (neither of them -> goalkeeper / referee / staff).
    """

    def __init__(self, vote_frames: int = 15, kit_tolerance: float = 2.0):
        self.team_colors = {}            # {1: bgr, 2: bgr}  (display colors)
        self.centers = None              # {1: lab, 2: lab}
        self.max_dist = None             # farther than this from every center -> kit 0
        self.votes = defaultdict(list)   # track key -> [1, 1, 0, 1 ...]
        self.kit = {}                    # track key -> final kit
        self.vote_frames = vote_frames
        self.kit_tolerance = kit_tolerance

    def get_player_color(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 5:  # height then width
            return None

        # Jersey area only: skip the head (top 15%) and the shorts/legs (bottom half)
        top_half = crop[int(crop.shape[0] * 0.15): crop.shape[0] // 2, :]
        pixels = top_half.reshape(-1, 3).astype(np.float32)

        # KMeans(k=2) needs at least 2 distinct colors
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

    def _colors(self, frames_and_boxes):
        colors = []
        for frame, boxes in frames_and_boxes:
            for bbox in boxes:
                c = self.get_player_color(frame, bbox)
                if c is not None:
                    colors.append(c)
        return np.array(colors)

    # ---------- learn the kits ---------- #
    def fit_team_colors(self, frames_and_boxes):
        """frames_and_boxes: list of (frame, [player bbox, ...])"""
        colors = self._colors(frames_and_boxes)
        if len(colors) < 4:
            raise RuntimeError("Not enough players detected to fit team colors.")

        feats = np.array([kit_feature(c) for c in colors])
        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0).fit(feats)

        # "how different can two shirts of the same team look" -> anything far beyond that is another kit
        spread = np.linalg.norm(feats - km.cluster_centers_[km.labels_], axis=1)
        self.max_dist = float(np.percentile(spread, 90) * self.kit_tolerance)
        self.centers = {1: km.cluster_centers_[0], 2: km.cluster_centers_[1]}

        raw = {t: colors[km.labels_ == t - 1].mean(axis=0) for t in (1, 2)}
        c1, c2 = self._display_color(raw[1]), self._display_color(raw[2])
        # if both teams map to the same display color, use raw colors instead
        if np.array_equal(c1, c2):
            c1, c2 = raw[1].astype(np.float32), raw[2].astype(np.float32)
        self.team_colors = {1: c1, 2: c2}

    def classify(self, color: np.ndarray):
        f = kit_feature(color)
        d = {k: float(np.linalg.norm(f - c)) for k, c in self.centers.items()}
        best = min(d, key=d.get)
        return best if d[best] <= self.max_dist else 0

    # ---------- kit of a tracked person (voted over several frames) ---------- #
    def get_kit(self, frame, bbox, key):
        if key in self.kit:
            return self.kit[key]
        if self.centers is None:  # kits were never fitted
            return None

        color = self.get_player_color(frame, bbox)
        if color is None:
            return self._current_vote(key)

        self.votes[key].append(self.classify(color))
        if len(self.votes[key]) >= self.vote_frames:
            self.kit[key] = Counter(self.votes[key]).most_common(1)[0][0]
            del self.votes[key]
            return self.kit[key]
        return self._current_vote(key)

    def is_locked(self, key) -> bool:
        return key in self.kit

    def _current_vote(self, key):
        v = self.votes.get(key)
        return Counter(v).most_common(1)[0][0] if v else None


# --------------------------------------------------------------------------- #
#                                 Drawing                                     #
# --------------------------------------------------------------------------- #
def draw_player(frame, bbox, color, label):
    x1, y1, x2, y2 = map(int, bbox)
    cx, w = (x1 + x2) // 2, x2 - x1
    color = tuple(int(c) for c in color)

    # 1. Dark outer border for contrast against green pitch
    cv2.ellipse(frame, (cx, y2), (int(w * 0.6), int(0.2 * w)), 0, -45, 235, (0, 0, 0), 4, cv2.LINE_AA)
    # 2. Main color ellipse
    cv2.ellipse(frame, (cx, y2), (int(w * 0.6), int(0.2 * w)), 0, -45, 235, color, 2, cv2.LINE_AA)

    # Label badge (track id, GK1, REF, STAFF ...)
    rw, rh = max(40, 13 * len(label) + 8), 20
    rx1, ry1 = cx - rw // 2, y2 + 10
    cv2.rectangle(frame, (rx1, ry1), (rx1 + rw, ry1 + rh), color, cv2.FILLED)
    cv2.rectangle(frame, (rx1, ry1), (rx1 + rw, ry1 + rh), (0, 0, 0), 1)

    txt_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(frame, label, (rx1 + 6, ry1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 2)


def draw_ball(frame, bbox):
    x1, y1, x2, _ = map(int, bbox)
    cx = (x1 + x2) // 2
    pts = np.array([[cx, y1 - 4], [cx - 9, y1 - 20], [cx + 9, y1 - 20]], np.int32)
    cv2.fillPoly(frame, [pts], ROLE_COLORS["ball"], cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)


def draw_legend(frame, rows):
    """rows: [(text, color), ...] -> small panel in the top-left corner."""
    x, y, rh = 15, 15, 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + 190, y + rh * len(rows) + 10), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (text, color) in enumerate(rows):
        cy = y + 18 + i * rh
        cv2.circle(frame, (x + 15, cy), 8, tuple(int(c) for c in color), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(frame, text, (x + 32, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = Config

    model = YOLO(path(args.model))
    print("Model classes:", model.names)
    person_model = YOLO(path(args.person_model)) if args.person_model else None

    # model class id -> role
    cls_role = {i: args.role_map[n.lower()] for i, n in model.names.items() if n.lower() in args.role_map}
    if not cls_role:
        raise ValueError(f"None of {list(args.role_map)} found in model classes: {list(model.names.values())}")

    cap = cv2.VideoCapture(path(args.source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(path(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    assigner = TeamAssigner(vote_frames=args.vote_frames, kit_tolerance=args.kit_tolerance)
    role_votes = defaultdict(Counter)    # best.pt track -> Counter of model classes (avoids class flicker)
    gk_team_votes = defaultdict(Counter) # goalkeeper track -> Counter of nearest team
    person_link = defaultdict(Counter)   # COCO person track -> Counter of matching best.pt track ids
    staff_votes = Counter()              # COCO person track -> frames it looked like staff
    last_state = {}                      # best.pt track -> (role, team) last time it was seen
    final_roles = {}                     # ("track"|"person", id) -> last role (for the summary)
    buffer = []
    fitted = False

    def detect(frame):
        """Returns ([(bbox, track_id, role, on_pitch), ...], [(bbox, person_id, on_pitch), ...])"""
        res = model.track(
            frame, persist=True, conf=args.conf, iou=args.nms_iou, tracker="bytetrack.yaml",
            classes=sorted(cls_role), verbose=False,
        )[0]
        poly = pitch_polygon(frame) if args.use_pitch_filter else None
        dets = []
        if res.boxes is not None and res.boxes.id is not None:
            for b, c, t in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.cls.cpu().numpy(),
                               res.boxes.id.cpu().numpy()):
                dets.append((b, int(t), cls_role[int(c)], is_on_pitch(poly, b)))

        persons = []
        if person_model is not None:
            res = person_model.track(
                frame, persist=True, conf=args.person_conf, imgsz=args.person_imgsz,
                tracker="bytetrack.yaml", classes=[0], verbose=False,
            )[0]
            if res.boxes is not None and res.boxes.id is not None:
                for b, t in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy()):
                    persons.append((b, int(t), is_on_pitch(poly, b)))
        return dets, persons

    def players_only(dets):
        return [b for b, _, r, on in dets if r == "player" and on]

    def fit(samples):
        assigner.fit_team_colors([(f, players_only(d)) for f, d, _ in samples])

    def resolve(frame, bbox, tid, raw_role, on_pitch):
        """Final (role, team) of a best.pt track: model votes + where it stands + which kit it wears."""
        votes = role_votes[tid]
        votes[raw_role] += 1
        gk_share = votes["goalkeeper"] / sum(votes.values())
        # the model calls the goalkeeper "player" most of the time -> a minority of GK votes is enough
        if votes["goalkeeper"] >= 2 and gk_share >= args.gk_min_share:
            role = "goalkeeper"
        else:
            role = votes.most_common(1)[0][0]

        if role in ("ball", "coach"):
            return role, None

        key = ("track", tid)
        kit = assigner.get_kit(frame, bbox, key)
        locked = assigner.is_locked(key)
        if role == "referee":
            # referees must wear colors different from both teams -> a "referee" in a team kit is a player
            if not (locked and kit in (1, 2)):
                return "referee", None
            role = "player"

        # substitutes / staff: player or goalkeeper standing outside the pitch
        if not on_pitch:
            return "staff", None
        if locked and kit == 0:
            return "goalkeeper", None             # on the pitch, but wears neither team's kit
        if role == "goalkeeper":
            if locked and kit in (1, 2) and gk_share < 0.5:
                return "player", kit              # a few wrong GK votes, but clearly a team kit
            return "goalkeeper", None
        return "player", kit if kit in (1, 2) else None

    def render(frame, dets, persons):
        items, balls, team_pts = [], [], defaultdict(list)

        # pass 1: best.pt tracks -> stable role + team
        for bbox, tid, raw_role, on_pitch in dets:
            role, team = resolve(frame, bbox, tid, raw_role, on_pitch)
            if role == "ball":
                balls.append(bbox)
                continue
            last_state[tid] = (role, team)
            items.append((bbox, tid, role, team))

        # pass 2: COCO persons that best.pt did not find -> staff (or a known person best.pt lost this frame)
        tracked = list(items)                     # best.pt tracks only (staff added below must not be linked to)
        drawn = {tid for _, tid, _, _ in items}
        for pbox, pid, on_pitch in persons:
            best_tid, best_iou = None, 0.0
            for bbox, tid, _, _ in tracked:
                iou, _ = overlap(pbox, bbox)
                if iou > best_iou:
                    best_tid, best_iou = tid, iou
            if best_iou >= 0.5:
                person_link[pid][best_tid] += 1   # same person as a best.pt track
            if any(overlap(pbox, bbox)[1] >= 0.5 for bbox, _, _, _ in tracked):
                continue                          # best.pt already has this person
            if not on_pitch:
                continue                          # crowd in the stands

            link = person_link[pid].most_common(1)
            if link and link[0][1] >= 3:
                tid = link[0][0]                  # known player / GK / referee that best.pt missed this frame
                if tid not in drawn and tid in last_state:
                    items.append((pbox, tid, *last_state[tid]))
                    drawn.add(tid)
                continue

            kit = assigner.get_kit(frame, pbox, ("person", pid))
            if kit in (1, 2):
                continue                          # wears a team kit: a player, not staff
            staff_votes[pid] += 1
            # the tracker sometimes gives one person two ids -> draw only one staff box
            already = any(overlap(pbox, bbox)[1] >= 0.5 for bbox, _, _, _ in items)
            if staff_votes[pid] >= args.staff_min_frames and not already:
                items.append((pbox, ("person", pid), "staff", None))

        for bbox, tid, role, team in items:
            if role == "player":
                team_pts[team].append(((bbox[0] + bbox[2]) / 2, bbox[3]))
        team_pts.pop(None, None)
        centroids = {t: np.mean(p, axis=0) for t, p in team_pts.items()}

        # pass 3: draw (goalkeeper's team = team whose players are, on average, closest)
        counts = Counter()
        for bbox, tid, role, team in items:
            if role == "player":
                color = assigner.team_colors.get(team, (200, 200, 200))
                label = str(tid)
                counts[f"team{team}" if team else "unknown"] += 1
            elif role == "goalkeeper":
                if centroids:
                    gx, gy = (bbox[0] + bbox[2]) / 2, bbox[3]
                    near = min(centroids, key=lambda t: np.hypot(gx - centroids[t][0], gy - centroids[t][1]))
                    gk_team_votes[tid][near] += 1
                gk_team = gk_team_votes[tid].most_common(1)[0][0] if gk_team_votes[tid] else None
                color = ROLE_COLORS["goalkeeper"]
                label = f"GK{gk_team}" if gk_team else "GK"
                counts["goalkeeper"] += 1
            else:
                color = ROLE_COLORS.get(role, (200, 200, 200))
                label = ROLE_LABELS.get(role, role.upper())
                counts[role] += 1
            final_roles[tid if isinstance(tid, tuple) else ("track", tid)] = role
            draw_player(frame, bbox, color, label)

        for bbox in balls:
            draw_ball(frame, bbox)

        rows = [(f"Team {t}: {counts[f'team{t}']}", assigner.team_colors[t]) for t in sorted(assigner.team_colors)]
        rows += [(f"Goalkeeper: {counts['goalkeeper']}", ROLE_COLORS["goalkeeper"]),
                 (f"Referee: {counts['referee']}", ROLE_COLORS["referee"]),
                 (f"Staff: {counts['staff']}", ROLE_COLORS["staff"])]
        draw_legend(frame, rows)

        writer.write(frame)
        if args.show:
            cv2.imshow("teams", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    stop, n_frames = False, 0
    while not stop:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n_frames >= args.max_frames):
            break
        n_frames += 1
        dets, persons = detect(frame)

        if not fitted:
            buffer.append((frame, dets, persons))
            if len(buffer) % args.fit_frames == 0:
                try:
                    fit(buffer)
                except RuntimeError:
                    continue  # not enough players yet -> keep buffering
                fitted = True
                print("Team colors (BGR):", {k: v.astype(int).tolist() for k, v in assigner.team_colors.items()})
                print(f"Kit distance limit: {assigner.max_dist:.1f}")
                for f, d, p in buffer:
                    if not render(f, d, p):
                        stop = True
                        break
                buffer.clear()
            continue

        if not render(frame, dets, persons):
            break

    if not fitted and buffer:
        try:
            fit(buffer)
        except RuntimeError:
            print("Warning: not enough players to learn team colors; drawing in gray.")
        for f, d, p in buffer:
            if not render(f, d, p):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Tracks per role:", dict(Counter(final_roles.values())))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
