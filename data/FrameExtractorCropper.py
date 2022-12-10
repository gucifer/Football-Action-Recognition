import cv2
import os


class FrameExtractorCropper():
	def save_frames(self, path, savePath, vidFileName, saveCount, fps):
		assert saveCount > 0
		assert saveCount <= fps
		video = cv2.VideoCapture(os.path.join(path, vidFileName))
		currentFrame = 0
		frameCount = 0
		framesToSave = []
		skipFrames = fps/saveCount
		for i in range(saveCount):
			framesToSave.append(1 + int(i * skipFrames))
		while(True):
			currentFrame += 1
			ret, frame = video.read()
			if ret:
				if currentFrame%fps in (framesToSave):
					frameCount += 1
					writeFile = os.path.join(savePath, str(frameCount) + '.jpg')
					os.makedirs(savePath, exist_ok=True)
					frame = self.crop_frames(frame=frame)
					write = cv2.imwrite(writeFile, frame)
			else:
				break
		video.release()
		cv2.destroyAllWindows()

	def crop_frames(self, frame):
		off_side = int((frame.shape[1] - 224)/2)
		frame = frame[:, off_side:-off_side, :]
		return frame

	def extract_crop_frames(self, vidFolder, frameFolder):
		leagues = os.listdir(vidFolder)
		for league in leagues:
			leagueWritePath = os.path.join(frameFolder, league)
			league = os.path.join(vidFolder, league)
			if not os.path.isdir(league): continue
			years = os.listdir(league)
			for year in years:
				yearWritePath = leagueWritePath + "_" + year
				year = os.path.join(league, year)
				if not os.path.isdir(year): continue
				matches = os.listdir(year)
				for match in matches:
					matchWritePath = yearWritePath + "_" + match
					match = os.path.join(year, match)
					if not os.path.isdir(match): continue
					vids = os.listdir(match)
					for vid in vids:
						if vid == ".DS_Store": continue
						vidPart = vid.split("_")[0]
						vidWritePath = matchWritePath + "_" + vidPart
						if os.path.exists(vidWritePath) and len(os.listdir(vidWritePath)) > 0:
							continue
						self.save_frames(match, vidWritePath, vid, 2, 25)
					print("Saved frames for {}".format(match))
