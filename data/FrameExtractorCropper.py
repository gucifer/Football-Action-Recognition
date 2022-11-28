import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool


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

def crop_frames(image):
	frame = cv2.imread(image)
	# frame = imutils.resize(frame, height=224)
	off_side = int((frame.shape[1] - 224)/2)
	frame = frame[:, off_side:-off_side, :]
	return frame

def list_vids_dirs(vidFolder, frameFolder):
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

def list_crop_frames_in_path(match):
	frames_path = "/Users/srv/Library/CloudStorage/OneDrive-SharedLibraries-GeorgiaInstituteofTechnology/Alphas - Documents/Project Data/frames"
	dest_frames_path = "/Users/srv/Documents/Git/GaTech/dl_cse_7643/final_project/code/data/frames_cropped_data"
	match_path = frames_path + "/" + match
	match_frames = os.listdir(match_path)
	print("Cropping {}".format(match))
	for match_frame in match_frames:
		cropped_frame = crop_frames(match_path + "/" + match_frame)
		write_dir = dest_frames_path + "/" + match
		try: os.makedirs(write_dir)
		except FileExistsError: pass
		write = cv2.imwrite(write_dir + "/" + match_frame, cropped_frame)
	print("Done {}".format(len(os.listdir(dest_frames_path))))

if __name__ == "__main__":
	# list_vids_dirs("videos", "frames")
	frames_path = "/Users/srv/Library/CloudStorage/OneDrive-SharedLibraries-GeorgiaInstituteofTechnology/Alphas - Documents/Project Data/frames"
	matches = os.listdir(frames_path)
	# for match in matches:
	# for i in tqdm(range(0, 100), total = len(matches), desc ="Completed"):
		# match = matches[i]
		
	with Pool(10) as image_cropper:
		image_cropper.map(list_crop_frames_in_path, matches)