import cv2
import os


def save_frames(path, savePath, vidFileName, saveCount, fps):
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
				try:
					os.makedirs(savePath)
				except FileExistsError:
					pass
				write = cv2.imwrite(writeFile, frame)
		else:
			break
	video.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	vidFolder = "videos"
	frameFolder = "frames"
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
					vidPart = vid.split("_")[0]
					vidWritePath = matchWritePath + "_" + vidPart
					save_frames(match, vidWritePath, vid, 2, 25)
				print("Saved frames for {}".format(match))

