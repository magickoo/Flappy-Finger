import cv2
import time
import numpy as np
import mediapipe as mp
import os
import random
import pandas as pd

# Create a class for hand detection using mediapipe
class HandDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

    def get_finger_pos(self, frame):
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = self.hands.process(frame_rgb)

        finger_landmarks = []

        # If hands are detected
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                landmarks_list = [(landmark.x, landmark.y) for landmark in landmarks.landmark]
                finger_landmarks.append(landmarks_list)

        return finger_landmarks


class Pipe:
    def __init__(self, x, y, width, gap=100, ymin=0, ymax=100):
        self.pipeup = PipeSegment(x, y + int(gap / 2), width, yend=ymax, type='up')
        self.pipedown = PipeSegment(x, y - int(gap / 2), width, yend=ymin, type='down')

    def move(self, dx):
        self.pipeup.move(dx)
        self.pipedown.move(dx)

    def __getitem__(self, index):
        return (self.pipeup, self.pipedown)[index]


class PipeSegment:
    def __init__(self, x, y, width, yend=100, type='up'):
        self.x = x
        self.y = y
        self.width = width
        self.yend = yend
        self.type = type
        self.startpoint = (int(x), int(y))
        self.endpoint = (int(x + width), int(yend))

    def move(self, dx):
        self.x -= dx
        self.startpoint = (int(self.x), int(self.y))
        self.endpoint = (int(self.x + self.width), int(self.yend))


def inside_box(x, y, box):
    return x > box[0] and y > box[1] and x < box[2] and y < box[3]


class FlappyBird:
    def __init__(self, detector, configfile='./flappybirdconfig', pipespacing=0.6, openingmode='startscreen'):

        self.detector = detector
        self.configfile = configfile
        self.pipespacing = pipespacing
        self.openingmode = openingmode
        self.cap = cv2.VideoCapture(0)

        self.background = cv2.imread(os.path.join(configfile, "background.png"))
        self.startscreen = cv2.imread(os.path.join(configfile, "startscreen.png"))
        self.gameoverscreen = cv2.imread(os.path.join(configfile, "gameover.png"))
        self.instructionsscreen = cv2.imread(os.path.join(configfile, "instructions.png"))
        self.playwindow_width = self.background.shape[1]
        self.playwindow_height = self.background.shape[0]
        self.pipes = []
        self.pipespacing = pipespacing
        self.pipexmax = self.playwindow_width
        self.pipewidth = 50
        self.birdsize = (50, 50)
        self.birdimage = cv2.resize(cv2.imread(os.path.join(configfile, "bird.png")), self.birdsize)
        self.start = False
        self.birdcoords = (0, 0, 0, 0)
        self.score = 0
        self.highscore = pd.read_csv(os.path.join(configfile, "highscore.csv"), index_col=0).values[0][0]
        self.mode = openingmode

    def _genpipes(self):
        spacing = self.pipespacing * self.playwindow_width
        while self.pipexmax + spacing < self.playwindow_width * 1.3:
            self.pipexmax += spacing
            newpipe = Pipe(self.pipexmax, random.randint(100, self.playwindow_height - 100), self.pipewidth,
                           ymin=0, ymax=self.playwindow_height)
            self.pipes.append(newpipe)

    def _drawpipes(self, img):
        for pipepair in self.pipes:
            for pipe in pipepair:
                if pipe.x >= 0 and pipe.x + pipe.width <= self.playwindow_width:
                    img = cv2.rectangle(img, pipe.startpoint, pipe.endpoint, (255, 0, 0), cv2.FILLED)
        return img

    def _update_bird_pos(self, img, x, y):

        ymin = int(y - self.birdsize[0] / 2)
        xmin = int(x - self.birdsize[1] / 2)

        if ymin < 0:
            ymin = 0
        if xmin < 0:
            xmin = 0
        if xmin + self.birdsize[1] > self.playwindow_width:
            xmin = self.playwindow_width - self.birdsize[1]
        if ymin + self.birdsize[0] > self.playwindow_height:
            ymin = self.playwindow_height - self.birdsize[0]

        self.birdcoords = [ymin, ymin + self.birdsize[1], xmin, xmin + self.birdsize[1]]
        img[ymin:ymin + self.birdsize[0], xmin:xmin + self.birdsize[1]] = self.birdimage
        return img

    def _detect_collision(self):
        for pipepair in self.pipes:
            for pipe in pipepair:
                x_condition = self.birdcoords[3] > pipe.x and self.birdcoords[2] < pipe.x + self.pipewidth
                y_condition_up = self.birdcoords[1] > pipe.y and pipe.type == 'up'
                y_condition_down = self.birdcoords[0] < pipe.y and pipe.type == 'down'
                if x_condition and (y_condition_up or y_condition_down):
                    return True
        return False

    def _show_cursor(self, img, x, y, alpha=0.1, counter=0):
        radius = 20
        selection_confirmed = False
        overlay = np.zeros_like(img)
        cv2.circle(overlay, (int(x), int(y)), radius, (255, 255, 255), cv2.FILLED)
        img = cv2.addWeighted(img, 1, overlay, alpha, 0)
        if counter > 0:
            overlay2 = np.zeros_like(img)
            cv2.circle(overlay2, (int(x), int(y)), min(radius, int(counter / 2)), (255, 255, 255), cv2.FILLED)
            img = cv2.addWeighted(img, 1, overlay2, 1, 0)
            if int(counter / 2) == radius:
                selection_confirmed = True
        return img, selection_confirmed

    def run(self):
        if self.mode == 'startscreen':
            self._run_start_screen()
        if self.mode == "showinstructions":
            self._run_instructions_screen()
        if self.mode == "playgame":
            self._run_game()
        if self.mode == 'gameover':
            self._run_start_screen()
            

    def _run_start_screen(self):

        playy, playx, pdy, pdx = self.startscreen.shape[0] * 0.57, self.startscreen.shape[1] * 0.27, 120, 80
        playbox = (int(playy), int(playx), int(playy + pdy), int(playx + pdx))
        insy, insx, idy, idx = self.startscreen.shape[0] * 0.52, self.startscreen.shape[1] * 0.42, 200, 80
        insbox = (int(insy), int(insx), int(insy + idy), int(insx + idx))

        playcounter = 0
        inscounter = 0
        while True:
            ret, frame = self.cap.read()
            finger_positions = self.detector.get_finger_pos(frame)
            if finger_positions:
                cursorx, cursory = finger_positions[0][8]
            else:
                cursorx, cursory = 0, 0
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = self._show_cursor(self.startscreen.copy(), x, y, alpha=0.1, counter=max(playcounter, inscounter))
            img = cv2.rectangle(img, playbox[0:2], playbox[2:], (255, 0, 0), 1)
            img = cv2.rectangle(img, insbox[0:2], insbox[2:], (0, 0, 255), 1)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            if inside_box(x, y, playbox):
                playcounter += 1
            else:
                playcounter = 0

            if inside_box(x, y, insbox):
                inscounter += 1
            else:
                inscounter = 0

            if playcounter > 0 and s:
                self.mode = 'playgame'
                break

            if inscounter > 0 and s:
                self.mode = 'showinstructions'
                break

        return

    def _run_instructions_screen(self):

        backy, backx, backdy, backdx = self.instructionsscreen.shape[0] * 0.55, self.startscreen.shape[1] * 0.47, 120, 80
        backbox = (int(backy), int(backx), int(backy + backdy), int(backx + backdx))
        backcounter = 0
        while True:
            ret, frame = self.cap.read()
            finger_positions = self.detector.get_finger_pos(frame)
            if finger_positions:
                cursorx, cursory = finger_positions[0][8]
            else:
                cursorx, cursory = 0, 0
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = self._show_cursor(self.instructionsscreen.copy(), x, y, alpha=0.1, counter=backcounter)
            img = cv2.rectangle(img, backbox[0:2], backbox[2:], (255, 0, 255), 1)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            if inside_box(x, y, backbox):
                backcounter += 1
            else:
                backcounter = 0

            if backcounter > 0 and s:
                self.mode = 'startscreen'
                break
        return

    def _run_countdown_screen(self):
        img = self.background.copy()
        font = cv2.FONT_HERSHEY_PLAIN
        fontscale = 6
        thickness = 2
        color = (255, 0, 255)
        (width, height), baseline = cv2.getTextSize("READY", font, fontscale, thickness)
        img = cv2.putText(img, "READY", (int(img.shape[1] / 2 - width / 2), int(img.shape[0] / 2 - height / 2)),
                          font, fontscale, color, thickness)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        time.sleep(1.5)
        img = self.background.copy()
        (width, height), baseline = cv2.getTextSize("GO!", font, fontscale, thickness)
        img = cv2.putText(img, "GO!", (int(img.shape[1] / 2 - width / 2), int(img.shape[0] / 2 - height / 2)),
                          font, fontscale, color, thickness)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        time.sleep(0.5)
        return

    def _run_game(self):
        self.__init__(self.detector, configfile=self.configfile, pipespacing=self.pipespacing, openingmode='playgame')
        collision = False
        self._run_countdown_screen()
        while not collision:
            collision = self._tick_game()
        self.mode = 'gameover'
        return

    def _tick_game(self):
        ret, frame = self.cap.read()
        finger_positions = self.detector.get_finger_pos(frame)
        if finger_positions:
            cursorx, cursory = finger_positions[0][8]
        else:
            cursorx, cursory = 0, 0
        x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
        dx = 10
        for pipepair in self.pipes:
            for pipe in pipepair:
                pipe.move(dx)
            if pipepair[0].x < 0:
                self.pipes.remove(pipepair)

        img = self.background.copy()
        self._genpipes()
        img = self._drawpipes(img)
        img = self._update_bird_pos(img, x, y)
        self.pipexmax += -dx
        collision = self._detect_collision()
        self.score += dx

        textx, texty = 20, 20
        scoretext = "score: {}".format(self.score)
        cv2.putText(img, scoretext, (textx, texty), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 1)
        (width, height), baseline = cv2.getTextSize(scoretext, cv2.FONT_HERSHEY_PLAIN, 2, 1)
        cv2.putText(img, "highscore: {}".format(self.highscore), (textx, texty + height + 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        return collision

    
    
            
    

    def _game_over_screen(self):
        img = self.gameoverscreen
        font = cv2.FONT_HERSHEY_PLAIN
        fontscale = 2
        thickness = 2
        color = (255, 0, 0)
        y = 300
        margin = 30
        retry_pressed = False  # Flag to track if the retry button is pressed

        if self.score > self.highscore:
            new_highscore = pd.DataFrame(index=['highscore'], columns=['value'], data=self.score)
            new_highscore.to_csv(os.path.join(self.configfile, 'highscore.csv'))
            self.highscore = self.score
            (width, height), baseline = cv2.getTextSize("NEW HIGH SCORE!".format(self.score), font, fontscale, thickness)
            img = cv2.putText(img, "NEW HIGH SCORE!", (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                              font, fontscale, color, thickness)
            y += height / 2 + margin

        (width, height), baseline = cv2.getTextSize("score {}".format(self.score), font, fontscale, thickness)
        img = cv2.putText(img, "score {}".format(self.score), (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                          font, fontscale, color, thickness)
        y += height / 2 + margin

        (width, height), baseline = cv2.getTextSize("highscore {}".format(self.highscore), font, fontscale, thickness)
        img = cv2.putText(img, "highscore {}".format(self.highscore),
                          (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                          font, fontscale, color, thickness)
        y += height / 2 + margin

        rety, retx, retdy, retdx = self.startscreen.shape[0] * 0.55, self.startscreen.shape[1] * 0.38, 150, 60
        retbox = (int(rety), int(retx), int(rety + retdy), int(retx + retdx))
        retry_counter = 0
        img_copy = cv2.rectangle(img, retbox[0:2], retbox[2:], (255, 0, 255), 1)

        while True:
            ret, frame = self.cap.read()
            finger_positions = self.detector.get_finger_pos(frame)
            if finger_positions:
                cursorx, cursory = finger_positions[0][8]
            else:
                cursorx, cursory = 0, 0
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = self._show_cursor(img_copy, x, y, alpha=0.1, counter=retry_counter)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

            if inside_box(x, y, retbox):
                retry_counter += 1
            else:
                retry_counter = 0

            if retry_counter > 0 and s:
                # Reset game state variables for a retry
                self.score = 0  # Reset the score
                self.pipes = []  # Reset the pipes
                self.mode = 'playgame'  # Set the mode to playgame to restart the game
                self._run_countdown_screen()  # Start the game again if retry is pressed
                break



# Create a HandDetector object
detector = HandDetector()


while True:  # Infinite loop to keep the game running
    # Create a FlappyBird object for each iteration
    game = FlappyBird(detector)

    # Game loop for the current game instance
    while True:
        ret, webcam_frame = game.cap.read()  # Read a frame from the webcam
        if game.mode == 'startscreen':
            game._run_start_screen()
        elif game.mode == "showinstructions":
            game._run_instructions_screen()
        elif game.mode == "playgame":
            game._run_game()
        elif game.mode == 'gameover':
            game._game_over_screen()

        

    # Release the OpenCV video capture for the current game instance
    game.cap.release()
    cv2.destroyAllWindows()

# Release the webcam video capture
cv2.destroyWindow("Webcam Feed")

# Release the webcam and close all OpenCV windows
#game.cap.release()
#cv2.destroyAllWindows()