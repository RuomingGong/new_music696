import sys
from PyQt5.QtWidgets import (QLineEdit, QLabel, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QApplication, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (QIntValidator,QDoubleValidator)

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from feature_ex import Feature_ex
from Guichu_Generator import Guichu_Generator
from graph_view import CMViewer


class Window(QWidget):

	def __init__(self):
		super().__init__()

		self.init_ui()

	def init_ui(self):
		self.CM = np.zeros((10,10))
		self.feature = None
		self.generator = None
		self.onlyDouble = QDoubleValidator()
		self.onlyInt = QIntValidator()

		self.b1 = QPushButton('Clear')
		self.b2 = QPushButton('Run')
		self.b3 = QPushButton('import')

		self.video_csv_source = QLineEdit() #path to video csv source
		self.video_audio_source = QLineEdit() #path to video csv source
		self.video_fps = QLineEdit() #path to video csv source
		self.video_fps.setValidator(self.onlyDouble)
		self.music_source = QLineEdit() #path to music midi source
		self.sample_rate = QLineEdit() #sample rate
		self.sample_rate.setValidator(self.onlyInt)
		
		self.csv_label = QLabel(self)
		self.audio_label = QLabel(self)
		self.fps_label = QLabel(self)
		self.ms_label = QLabel(self)
		self.sr_label = QLabel(self)
		
		self.csv_label.setText('video_csv_source')
		self.audio_label.setText('video_audio_source')
		self.fps_label.setText('video_fps')
		self.ms_label.setText('music_source')
		self.sr_label.setText('sample_rate')
		
		self.timeLabel = QLabel(self)
		self.timeLabel.setText('time weights:')
		self.l1 = QLineEdit()
		self.s1 = QSlider(Qt.Horizontal)
		self.s1.setMinimum(1)
		self.s1.setMaximum(99)
		self.s1.setValue(25)
		self.s1.setTickInterval(10)
		self.s1.setTickPosition(QSlider.TicksBelow)

		self.beatLabel = QLabel(self)
		self.beatLabel.setText('beat weights:')
		self.l2 = QLineEdit()
		self.s2 = QSlider(Qt.Horizontal)
		self.s2.setMinimum(1)
		self.s2.setMaximum(99)
		self.s2.setValue(25)
		self.s2.setTickInterval(10)
		self.s2.setTickPosition(QSlider.TicksBelow)

		self.motionLabel = QLabel(self)
		self.motionLabel.setText('motion weights:')
		self.l3 = QLineEdit()
		self.s3 = QSlider(Qt.Horizontal)
		self.s3.setMinimum(1)
		self.s3.setMaximum(99)
		self.s3.setValue(25)
		self.s3.setTickInterval(10)
		self.s3.setTickPosition(QSlider.TicksBelow)

		self.jumpLabel = QLabel(self)
		self.jumpLabel.setText('jump threshold:')
		self.l4 = QLineEdit()
		self.s4 = QSlider(Qt.Horizontal)
		self.s4.setMinimum(1)
		self.s4.setMaximum(99)
		self.s4.setValue(25)
		self.s4.setTickInterval(10)
		self.s4.setTickPosition(QSlider.TicksBelow)

		self.rateLabel = QLabel(self)
		self.rateLabel.setText('jump rate:')
		self.l5 = QLineEdit()
		self.s5 = QSlider(Qt.Horizontal)
		self.s5.setMinimum(1)
		self.s5.setMaximum(99)
		self.s5.setValue(25)
		self.s5.setTickInterval(10)
		self.s5.setTickPosition(QSlider.TicksBelow)

		self.selectLabel = QLabel(self)
		self.selectLabel.setText('path select rate:')
		self.l6 = QLineEdit()
		self.s6 = QSlider(Qt.Horizontal)
		self.s6.setMinimum(1)
		self.s6.setMaximum(99)
		self.s6.setValue(25)
		self.s6.setTickInterval(10)
		self.s6.setTickPosition(QSlider.TicksBelow)

		self.blockLabel = QLabel(self)
		self.blockLabel.setText('block size:')
		self.l7 = QLineEdit()
		self.s7 = QSlider(Qt.Horizontal)
		self.s7.setMinimum(1)
		self.s7.setMaximum(99)
		self.s7.setValue(25)
		self.s7.setTickInterval(10)
		self.s7.setTickPosition(QSlider.TicksBelow)

		v1_box = QVBoxLayout()
		v1_box.addWidget(self.csv_label)
		v1_box.addWidget(self.audio_label)
		v1_box.addWidget(self.fps_label)
		v1_box.addWidget(self.ms_label)
		v1_box.addWidget(self.sr_label)
		v1_box.addWidget(self.timeLabel)
		v1_box.addWidget(self.beatLabel)
		v1_box.addWidget(self.motionLabel)
		v1_box.addWidget(self.jumpLabel)
		v1_box.addWidget(self.rateLabel)
		v1_box.addWidget(self.selectLabel)
		v1_box.addWidget(self.blockLabel)


		v2_box = QVBoxLayout()
		v2_box.addWidget(self.video_csv_source)
		v2_box.addWidget(self.video_audio_source)
		v2_box.addWidget(self.video_fps)
		v2_box.addWidget(self.music_source)
		v2_box.addWidget(self.sample_rate)
		v2_box.addWidget(self.l1)
		v2_box.addWidget(self.l2)
		v2_box.addWidget(self.l3)
		v2_box.addWidget(self.l4)
		v2_box.addWidget(self.l5)
		v2_box.addWidget(self.l6)
		v2_box.addWidget(self.l7)

		h_box = QHBoxLayout()
		h_box.addLayout(v1_box)
		h_box.addLayout(v2_box)

		v_box = QVBoxLayout()
		v_box.addLayout(h_box)
		v_box.addWidget(self.b1)
		v_box.addWidget(self.b2)
		v_box.addWidget(self.b3)
		v_box.addWidget(self.s1)
		v_box.addWidget(self.s2)
		v_box.addWidget(self.s3)
		v_box.addWidget(self.s4)
		v_box.addWidget(self.s5)
		v_box.addWidget(self.s6)
		v_box.addWidget(self.s7)

		self.mainHBOX_param_scene = QHBoxLayout()

		#CM view
		layout_plot = QHBoxLayout()
		self.loaded_plot = CMViewer(self)
		#self.loaded_plot.setMinimumHeight(200)
		self.loaded_plot.update()
		layout_plot.addWidget(self.loaded_plot)

		self.mainHBOX_param_scene.addLayout(layout_plot)

		h_main_box = QHBoxLayout()
		h_main_box.addLayout(v_box)
		h_main_box.addLayout(self.mainHBOX_param_scene)
		
		self.setLayout(h_main_box)
		self.setWindowTitle('Music696')

		self.b1.clicked.connect(lambda: self.btn_clk(self.b1, 'Hello from Clear'))
		self.b2.clicked.connect(lambda: self.btn_clk(self.b2, 'Hello from Run'))
		self.b3.clicked.connect(lambda: self.btn_clk(self.b3, 'Hello from Import'))
		self.s1.valueChanged.connect(self.time_coef_change)
		self.s2.valueChanged.connect(self.beat_coef_change)
		self.s3.valueChanged.connect(self.motion_coef_change)
		self.s4.valueChanged.connect(self.jump_threshold_change)
		self.s5.valueChanged.connect(self.jump_rate_change)
		self.s6.valueChanged.connect(self.path_select_change)
		self.s7.valueChanged.connect(self.block_size_change)

		self.show()

	def btn_clk(self, b, string):
		if b.text() == 'Run':
			music_feature = self.feature.music_feature
			video_feature = self.feature.video_feature
			if (self.l1.text() == ""):
				time_coef = 100
			else:
				time_coef = float(self.l1.text())

			if (self.l2.text() == ""):
				beat_coef = 1
			else:
				beat_coef = float(self.l2.text())

			if (self.l3.text() == ""):
				motion_coef = 1
			else:
				motion_coef = float(self.l3.text())

			if (self.l4.text() == ""):
				jump_threshold = 5
			else:
				jump_threshold = float(self.l4.text())

			if (self.l5.text() == ""):
				jump_rate = 10
			else:
				jump_rate = float(self.l5.text())

			if (self.l6.text() == ""):
				path_select = 0.3
			else:
				path_select = float(self.l6.text())

			if (self.l7.text() == ""):
				block_size = 32
			else:
				block_size = int(self.l7.text())

			self.generator = Guichu_Generator(music_feature,video_feature,time_coef=time_coef,beat_coef=beat_coef,motion_coef=motion_coef,jump_threshold=jump_threshold,jump_rate=jump_rate,path_select=path_select,block_size=block_size)
			self.generator.DTW()

			self.CM = self.generator.matrix
			#######################################

			#CM view
			self.loaded_plot.update()
			print("alignment done!")
			
		elif b.text() == 'import':
			midi_file = self.music_source.text()
			csv_file = self.video_csv_source.text()
			audio_file = self.video_audio_source.text()
			video_fps = float(self.video_fps.text())
			if self.sample_rate.text() == "":
				sr = 10
			else:
				sr = int(self.sample_rate.text())
			self.feature = Feature_ex(midi_file,csv_file,audio_file,video_fps,sr=sr)
			print("import done!")

		else:
			self.video_source.clear()
			self.music_source.clear()
			self.l1.clear()
			self.l2.clear()
			self.l3.clear()
			self.l4.clear()
			self.l5.clear()
			self.l6.clear()
			self.l7.clear()
		print(string)

	def time_coef_change(self):
		time_coef = str(self.s1.value())
		self.l1.setText(time_coef)

	def beat_coef_change(self):
		beat_coef = str(self.s2.value())
		self.l2.setText(beat_coef)

	def motion_coef_change(self):
		motion_coef = str(self.s3.value())
		self.l3.setText(motion_coef)

	def jump_threshold_change(self):
		threshold = str(self.s4.value())
		self.l4.setText(threshold)

	def jump_rate_change(self):
		jump_rate = str(self.s5.value())
		self.l5.setText(jump_rate)

	def path_select_change(self):
		select_rate = str(self.s6.value())
		self.l6.setText(select_rate)

	def block_size_change(self):
		block_size = str(self.s7.value())
		self.l7.setText(block_size)


app = QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())