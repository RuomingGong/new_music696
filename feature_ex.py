import os
import copy as cp
from pylab import *
import pretty_midi
import librosa             # The librosa library
import librosa.display     # librosa's display module (for plotting features)
import IPython.display     # IPython's display module (for in-line audio)
import matplotlib.pyplot as plt # matplotlib plotting functions
import matplotlib.style as ms   # plotting style
import numpy as np              # numpy numerical functions
import scipy
import math
import random
import csv
import copy

class Feature_ex():

	def __init__(self,music_midi,video_csv,video_wav,video_fps,sr=5):
		super().__init__()
		self.init_gen(music_midi,video_csv,video_wav,video_fps,sr=sr)

	def init_gen(self,music_midi,video_csv,video_wav,video_fps,sr=5):
		self.music_midi = music_midi #midi file
		infile = open(video_csv,'r')
		self.video_csv = infile.read().strip().split('\n')
		self.video_wav = video_wav #audio from video
		self.midi_data = pretty_midi.PrettyMIDI(self.music_midi)
		self.video_fps = video_fps
		self.sr = sr
		self.dis_contour = self.pitch_contour() #discrete pitch contour
		self.cts_contour = self.build_contour() #cts pitch contour after go through gaussian filter
		self.beat_dis_contour = self.vel_contour() #discrete beat contour
		self.beat_cts_contour = self.build_beat_contour() #cts beat contour after go through gaussian filter
		self.video_beat_contour = self.build_beat_contour(pattern=True) #cts beat contour for video
		self.video_motion = self.motion_graph()
		self.video_feature = self.video_feature_fetch()
		self.music_feature = self.music_feature_fetch()


	#get beats of midi file
	def getbeat(self,midi_data):
		beats = midi_data.get_beats()
		return beats

	def show_score(self,S, fs = 100):
		imshow(S, aspect='auto', origin='bottom', interpolation='nearest', cmap=cm.gray_r)
		xlabel('Time')
		ylabel('Pitch')
		pc=array(['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'])
		idx = tile([0,4,7],13)[:128]
		yticks(arange(0,128,4),pc[idx], fontsize=5)
		xticks(arange(0,S.shape[1],fs),arange(0,S.shape[1],fs)/fs, ) 

	#return the pitch contour (time seires of highest pitch) of a midi file
	def pitch_contour(self):
		piano_roll = self.midi_data.get_piano_roll(fs=100)
		time_length = len(piano_roll[0])
		contour = np.zeros((128,time_length))
		for i in range(time_length):
			for j in range(128):
				if piano_roll[127-j][i] != 0:
					contour[127-j][i] = 1
					break
		#show_score(piano_roll)
		#self.show_score(contour)
		return contour

	#return the velocity of pitch contour，the contour is of size (time_length,)
	def vel_contour(self):
		piano_roll = self.midi_data.get_piano_roll(fs=100)
		time_length = len(piano_roll[0])
		contour = np.zeros((128,time_length))
		onset_contour = np.zeros((time_length,))
		for i in range(time_length):
			for j in range(128):
				if piano_roll[127-j][i] != 0:
					contour[127-j][i] = piano_roll[127-j][i]
					break
		for j in range(128):
			if contour[127-j][0]!=0:
				onset_contour[0] = contour[127-j][0]
		for i in range(1,time_length):
			for j in range(128):
				if contour[127-j][i] != 0 and contour[127-j][i-1] == 0:
					onset_contour[i] = contour[127-j][i]
		return onset_contour/400

	#construct a smooth pitch contour using gaussian weight 
	def gaussian(self,x, mu, sig):
		return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

	def normalize(self,l):
		temp = np.array(l)
		return temp/sum(temp)

	def convolution(self,p,temp_vec, window, Is_pitch=True):
		if p == 0 and Is_pitch:
			return 0
		result = 0
		weight_sum = 0
		for i in range(len(temp_vec)):
			if Is_pitch:
				if temp_vec[i] != 0:
					result = result + temp_vec[i]*window[i]
					weight_sum = weight_sum + window[i]
			else:
				result = result + temp_vec[i]*window[i]
				#print(result)
				weight_sum = 1
				#print(weight_sum)
		return result/weight_sum

	#create the continuous pitch contour
	def build_contour(self):
		time_length = len(self.dis_contour[0])
		#print(time_length)
		pitch_series = []
		for i in range(time_length):
			judge = True
			for j in range(128):
				if self.dis_contour[j][i] != 0:
					pitch_series.append(j)
					judge = False
					break
			if judge:
				pitch_series.append(0)
	    
		window_size = 201
		w_x = np.linspace(-100.0,100.0,num=window_size)
		w_y = []
		for i in range(window_size):
			w_y.append(self.gaussian(w_x[i],0,100))
		w_y = self.normalize(w_y)
	    
		x_axis = np.arange(time_length)
		y_axis = []
	    
		cts_contour = np.zeros((128,time_length))   
		print(time_length)
		for i in range(time_length):
			if i < int(window_size/2)+1:
				temp_vec = np.array(pitch_series[:i+101])
				temp_pitch = self.convolution(pitch_series[i],temp_vec,np.array(w_y[100-i:]))
				esti_pitch = int(round(temp_pitch))
				if temp_pitch != 0:
					cts_contour[esti_pitch][i] = 1
					y_axis.append(temp_pitch)
				else:
					if len(y_axis) == 0:
						y_axis.append(0)
					else:
						y_axis.append(y_axis[-1])
			elif i > time_length - int(window_size/2) - 1:
				temp_vec = np.array(pitch_series[i-100:time_length])
				temp_pitch = self.convolution(pitch_series[i],temp_vec,np.array(w_y[:103+time_length-i]))
				esti_pitch = int(round(temp_pitch))
				if temp_pitch != 0:
					cts_contour[esti_pitch][i] = 1
					y_axis.append(temp_pitch)
				else:
					y_axis.append(y_axis[-1])
			else:
				temp_vec = np.array(pitch_series)
				temp_pitch = self.convolution(pitch_series[i],temp_vec[i-100:i+101],w_y)
				esti_pitch = int(round(temp_pitch))
				if temp_pitch != 0:
					cts_contour[esti_pitch][i] = 1
					y_axis.append(temp_pitch)
				else:
					y_axis.append(y_axis[-1])
		#plt.plot(x_axis,y_axis)
		return y_axis

	#create the continuous beat_contour, set pattern to be true when deal with wav file 
	def build_beat_contour(self, pattern=False):
		if pattern:
			y, sr = librosa.load(self.video_wav)
			tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
			onset_env = librosa.onset.onset_strength(y, sr=sr,aggregate=np.median)
			hop_length = 512
			times = librosa.frames_to_time(np.arange(len(onset_env)),sr=sr, hop_length=hop_length)
			clicks = librosa.clicks(frames=beats, sr=sr, length=len(y))
			click_times = librosa.frames_to_time(beats,sr=sr, hop_length=hop_length)
			time_length = int(click_times[-1]*100)
			dis_contour = np.zeros((time_length,))
			for b in click_times:
				ind = int(b*100)
				if ind < time_length:
					dis_contour[ind] = dis_contour[ind] + 1
	    
		else:
			tick = self.getbeat(self.midi_data)
			#add note onset to contour
			#dis_contour = np.zeros((5560,))
			dis_contour = self.vel_contour()
			time_length = len(dis_contour)
			#add significant tempo to contour
			for b in tick:
				ind = int(b*100)
				if ind < time_length:
					dis_contour[ind] = dis_contour[ind] + 1
	    
	    
		pitch_series = dis_contour

		window_size = 201
		w_x = np.linspace(-100.0,100.0,num=window_size)
		w_y = []
		for i in range(window_size):
			w_y.append(self.gaussian(w_x[i],0,5))
		#print(w_y)
		#w_y = normalize(w_y)
		#print(w_y)

		x_axis = np.arange(time_length)
		y_axis = []

		cts_contour = np.zeros((128,time_length))   
		#print(time_length)
		for i in range(time_length):
			if i < int(window_size/2)+1:
				temp_vec = np.array(pitch_series[:i+101])
				temp_vel = self.convolution(pitch_series[i],temp_vec,np.array(w_y[100-i:]),Is_pitch=False)
				y_axis.append(temp_vel)
			elif i > time_length - int(window_size/2) - 1:
				temp_vec = np.array(pitch_series[i-100:time_length])
				temp_vel = self.convolution(pitch_series[i],temp_vec,np.array(w_y[:103+time_length-i]),Is_pitch=False)
				y_axis.append(temp_vel)
			else:
				temp_vec = np.array(pitch_series)
				temp_vel = self.convolution(pitch_series[i],temp_vec[i-100:i+101],w_y,Is_pitch=False)
				y_axis.append(temp_vel)
		p_x = []
		p_y = []
		for i in range(time_length):
			if y_axis[i] != 0:
				p_x.append(x_axis[i])
				p_y.append(y_axis[i])
		#plt.scatter(p_x[100:200],p_y[100:200])
		return y_axis

	#normalize height and time_length for l1
	#height is normalized to 0-100, time_length is normalized wrt the same time interval 1/fs0 s
	def norm_motion(self,l1,fs1,fs0=5):
		h = 100
		l1_time = int(len(l1)*fs0/fs1)+1
		m1 = []

		count = 0
		for i in range(l1_time-2):
			temp_height = 0
			local_count = 0
			while count <= (i+0.5)*fs1/fs0 and count > (i-0.5)*fs1/fs0:
				temp_height = temp_height + l1[count]
				count = count + 1
				local_count = local_count + 1
			temp_height = float(temp_height)
			m1.append(temp_height)
				
		l1_height_diff = max(m1) - min(m1)

		m1 = np.array(m1)
		m1 = m1*h/l1_height_diff

		return m1

	def get_beat_feature(self,beat_l,feature_pt_l):
		result_l = []
		#print(len(beat_l))
		#print(len(feature_pt_l))
		for i in range(len(feature_pt_l)-1):
			result_l.append([beat_l[int(feature_pt_l[i]*100)]])
		return result_l

	def list_combine(self,l1,l2):
		l3 = copy.deepcopy(l1)
		for i in range(len(l2)):
			l3[i].append(l2[i])
		return l3

	def pitch_diff(self,m):
		pitch_d = []
		for i in range(len(m)-1):
			pitch_d.append(m[i+1]-m[i])
		return pitch_d

	#get graph of motion wrt y_motion, return graph matrix
	def motion_graph(self):
		motion_data = [a.strip().split(',') for a in self.video_csv]
		y_motion = []
		for i in range(1,len(motion_data)-1):
			#print(i)
			y_motion.append(float(motion_data[i][7]))
		#print(y_motion)
		x_axis = np.arange(len(y_motion))
		#plt.scatter(x_axis,y_motion)
		return y_motion

	def music_feature_fetch(self):
		f = 1.0/self.sr
		#print("cts_contour",len(cts_contour))
		l1 = self.pitch_diff(self.cts_contour)
		#print("l1",len(l1))
		l1 = self.norm_motion(l1,100,fs0=self.sr)
		#print(len(l1))
		sample_list1 = []
		for i in range(int(len(self.beat_cts_contour)/(f*100))+1):
			sample_list1.append(f*i)
		music_feature = self.get_beat_feature(self.beat_cts_contour,sample_list1)
		#print(music_feature)
		for i in range(len(l1)):
			music_feature[i].append(l1[i])
			music_feature[i].insert(0,f*i)
		music_feature = music_feature[:len(l1)]
		return music_feature

	#we first convert wav to midi, the audifile we use is midi file, sr is sample rate for feature point,
	#fps is frame per second for motionfile
	def video_feature_fetch(self):
		f = 1.0/self.sr
		#print("l2",len(l2))
		l2 = self.norm_motion(self.video_motion,self.video_fps,fs0=self.sr)
		#print(len(l2))
		#y, sr = librosa.load(audiofile)
		#tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
		#onset_env = librosa.onset.onset_strength(y, sr=sr,aggregate=np.median)
		#hop_length = 512
		#times = librosa.frames_to_time(np.arange(len(onset_env)),sr=sr, hop_length=hop_length)
		#clicks = librosa.clicks(frames=beats, sr=sr, length=len(y))
		#click_times = librosa.frames_to_time(beats,sr=sr, hop_length=hop_length)
		sample_list2 = []
		for i in range(int(len(self.video_beat_contour)/(f*100))+1):
			sample_list2.append(f*i)
		video_feature = self.get_beat_feature(self.video_beat_contour,sample_list2)
		for i in range(min(len(l2),len(video_feature))):
			video_feature[i].append(l2[i])
			video_feature[i].insert(0,f*i)
		video_feature = video_feature[:min(len(l2),len(video_feature))]
		return video_feature





