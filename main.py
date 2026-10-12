from collections import Counter , defaultdict
import cv2
import numpy as np
from sklearn.cluster import KMeans
import ultralytics import YOLO


class Config:
    source="video.mp4"
    output="output.mp4"
    model="yolov8s.pt"
    player_classes=["person"] # i can customize the classes i want to detect for example['player','goalkeeper','referee']
    conf=0.3
    fit_frames=10
    vote_frames=15
    show=True

    

class TeamAssigner:
    def __init__(self, vote_frames: int = 15):
        self.team_colors = {} # {1: (B,G,R), 2: (B,G,R)}
        self.kmeans = None
        self.votes=defaultdict(list) # track_id -> list of team votes example {1: [1, 0, 1,1,1,1,0,1],2: [0, 1, 0,0,0,0,1,0]}
        self.player_team={}
        self.vote_frames=vote_frames

        def get_player_color(self, frame:np.ndarray, bbox)-> np.ndarray|None:
            x1,y1,x2,y2=map(int,bbox)
            h,w=frame.shape[:2]
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 5: # height then width
                return None

             # Use only the top half (jersey area, not shorts/legs)
            top_half = crop[: crop.shape[0] // 2, :] # (40,30,3)
            pixels = top_half.reshape(-1, 3).astype(np.float32) # (1200,3)
            km = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0).fit(pixels)
            labels = km.labels_.reshape(top_half.shape[0], top_half.shape[1])

            # Background cluster = the one that owns the majority of the 4 corners
            corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
            bg_cluster = max(set(corners), key=corners.count) # {0,1}
            player_cluster = 1 - bg_cluster
            return km.cluster_centers_[player_cluster]  

        def fit_team_colors(self,frames_and_boxes):
            colors=[]
            for frame,boxes in frames_and_boxes:
                for box in boxes:
                    color=self.get_player_color(frame,box)
                    if color is not None:
                        colors.append(color)

            if len(colors)<4:
                print(f'Not enough colors to fit team colors')

            colors=np.array(colors)

            km_initial=KMeans(n_clusters=3, init="k-means++", n_init=10,random_state=0).fit(colors)
            counts=np.bincount(km_initial.labels_,minlength=3) # (0,0,0,1,1,1,2,1,0,2,1,)

            top_two=np.argsort(counts)[-2:] # [1,2]
            team1_center=km_initial.cluster_centers_[top_two[0]]
            team2_center=km_initial.cluster_centers_[top_two[1]]

            for team_id,center in [(1,team1_center),(2,team2_center)]:
                b,g,r=center
                # Green dominant -> vibrant neon green
                if g>b+25 and g>r+15:
                    self.team_colors[team_id]=np.array([40,245,60],dtype=np.float32)
                # High brightness all channels -> crisp pure white
                elif min(b,g,r)>170:
                    self.team_colors[team_id]=np.array([255,255,255],dtype=np.float32)
                elif r>g+30 and r>b+30:
                    self.team_colors[team_id]=np.array([0,0,255],dtype=np.float32)
                elif b>g+30 and b>r+30:
                    self.team_colors[team_id]=np.array([255,80,0],dtype=np.float32)
                else:
                    self.team_colors[team_id]=center.astype(np.float32) # row detected color

            self.kmeans=KMeans(n_clusters=2,init=np.array[team1_center,team2_center],n_init=1,random_state=0)
            self.kmeans.fit(np.array([team1_center,team2_center]))





def main():
    args=Config

    model=YOLO(args.model)
    player_cls_ids={i for i,n in model.names.items() if n in args.player_classes}
    if not player_cls_ids:
        print(f"None of {args.player_classes} found in model classes: {list(model.names.values())}")

    cap=cv2.VideoCapture(args.source)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    W,H=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))

    assigner=TeamAssigner(vote_frames=args.vote_frames)
    buffer=[]
    fitted=False