import sys 
import threading
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np
import random
import time
import logging

#Canvas
canvasWidth, canvasHeight = 500, 500

#Camera
cameraPos = [50, 50, -50]
screenWidth, screenHeight = 100, 100
image = np.zeros((screenWidth, screenHeight, 4), dtype=np.int)

#Particle System
numOfParticles = 100
ColorType = 1
systemSize = [100, 100, 100]
systemPos = [50, 50, 50]
systemXBounds = [systemPos[0] - systemSize[0] / 2, systemPos[0] + systemSize[0] / 2]
systemYBounds = [systemPos[1] - systemSize[1] / 2, systemPos[1] + systemSize[1] / 2]
systemZBounds = [systemPos[2] - systemSize[2] / 2, systemPos[2] + systemSize[2] / 2]
# x, y, z, Idx, Idy
Rays = []
# x, y, z, vx, vy, vz, [r, g, b]
Particles = []

#Ray Threading
numOfRaycastThreads = 4
raycastThreads = []
raycastThreadStartEvents = []
raycastThreadEndEvents = []

#Update Threading
numOfUpdateThreads = 4
updateThreads = []
updateThreadStartEvents = []
updateThreadEndEvents = []

#Gravity Threading
gravity = False
gravityStartEvent = threading.Event()
gravityEndEvent = threading.Event()


#Glut StartEvent
glutStartEvent = threading.Event()
glutEndEvent = threading.Event()

def generateParticles():
	logging.info("Generating Particles Started")
	for _ in range(numOfParticles):
		Particles.append(
			[
				random.uniform(systemXBounds[0], systemXBounds[1]),  #0 x
				random.uniform(systemYBounds[0], systemYBounds[1]),  #1 y
				random.uniform(systemZBounds[0], systemZBounds[1]),  #2 z
				random.uniform(-5,5), 																 #3 vx
				random.uniform(-5,5), 																 #4 vy
				random.uniform(-5,5), 																 #5 vz
				[
					random.randrange(0,255), 
					random.randrange(0,255),
					random.randrange(0,255)
				]
			] 
		)
	logging.info("Generating Particles Finnished")

	logging.info("Generating Update Threads Started")
	indexes = 0
	for _ in range(numOfRaycastThreads):
		startEvent = threading.Event()
		endEvent = threading.Event()
		thread = threading.Thread(target=updateSimulation, args=(indexes, indexes + int((numOfParticles / numOfUpdateThreads)), startEvent, endEvent))
		updateThreadStartEvents.append(startEvent)
		updateThreadEndEvents.append(endEvent)
		updateThreads.append(thread)
		indexes += int(numOfParticles / numOfUpdateThreads)
		thread.start()
	logging.info("Generating Update Threads Finnished")

def generateRays():
	logging.info("Generating Rays Started")
	for i in range(screenWidth):
		for j in range(screenHeight):
			Rays.append([(i + 0.5) - cameraPos[0], (j + 0.5) - cameraPos[1], cameraPos[2], i, j])
	pass
	logging.info("Generating Rays Finnished")

	logging.info("Generating Ray Threads Started")
	indexes = 0
	for i in range(numOfRaycastThreads):
		startEvent = threading.Event()
		endEvent = threading.Event()
		thread = threading.Thread(target=render, args=(indexes, indexes + int(screenHeight * screenWidth / numOfRaycastThreads), startEvent, endEvent))
		raycastThreadStartEvents.append(startEvent)
		raycastThreadEndEvents.append(endEvent)
		raycastThreads.append(thread)
		indexes += int(screenHeight * screenWidth / numOfRaycastThreads)
		thread.start()
	logging.info("Generating Ray Threads Finnished")

def NewColors():
	for particle in Particles:
		particle[6][0] = random.randrange(0,255)
		particle[6][1] = random.randrange(0,255)
		particle[6][2] = random.randrange(0,255)

def updateSimulation(startIndex, endIndex, startEvent, endEvent):
	centerX, centerY, centerZ = 0, 0, 0
	while True:
		startEvent.wait()
		logging.info("Update Thread started From Index: " + str(startIndex))
		startEvent.clear()
		if ColorType == 4:
			centerX = 0
			centerY = 0
			centerZ = 0
			for p in Particles:
				centerX = centerX + p[0]
				centerY = centerY + p[1]
				centerZ = centerZ + p[2]
				pass
			centerX = centerX / numOfParticles
			centerY = centerY / numOfParticles
			centerZ = centerZ / numOfParticles
			pass
		for p1 in range(startIndex, endIndex):
			for p2 in Particles:
				if Particles[p1] == p2: continue
				if (Particles[p1][0] - p2[0])**2 + (Particles[p1][1] - p2[1])**2  + (Particles[p1][2] - p2[2])**2 < 2:
						v1 = np.array((Particles[p1][3], Particles[p1][4], Particles[p1][5]))
						v2 = np.array((p2[3], p2[4], p2[5]))
						d = np.array((Particles[p1][0], Particles[p1][1], Particles[p1][2])) - np.array((p2[0], p2[1], p2[2]))
						n = d / np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
						u1 = v1 + np.dot(np.dot(v2 - v1, n), n)
						u2 = v2 + np.dot(np.dot(v1 - v2, n), n)
						Particles[p1][3] = u1[0]
						Particles[p1][4] = u1[1]
						Particles[p1][5] = u1[2]
						p2[3] = u2[0]
						p2[4] = u2[1]
						p2[5] = u2[2]

		for i in range(startIndex, endIndex):
			if ColorType == 1:
				pass
			elif ColorType == 2:
				#Speed
				Particles[i][6][0] = np.interp(np.sqrt(pow(Particles[i][3], 2) + pow(Particles[i][4], 2)+ pow(Particles[i][5], 2)), [0, 9], [0, 255])
				Particles[i][6][1] = 0
				Particles[i][6][2] = 0
				pass
			elif ColorType == 3:
				#Local force
				Particles[i][6][0] = 0
				Particles[i][6][1] = 0
				Particles[i][6][2] = 0
				for p in Particles:
					if np.sqrt(pow(Particles[i][0] -p[0], 2) + pow(Particles[i][1] -p[1], 2) + pow(Particles[i][2] -p[2], 2)) < 30:
						Particles[i][6][1]  += 1
				pass
				Particles[i][6][1] = np.interp(Particles[i][6][1] , [0, numOfParticles], [0, 255])
			else:
				#Center of Mass
				Particles[i][6][0] = 0
				Particles[i][6][1] = 0
				Particles[i][6][2] = np.interp(np.sqrt(pow(Particles[i][0] - centerX, 2)+ pow(Particles[i][1] - centerY, 2) + pow(Particles[i][2] - centerZ, 2)), [0, 100], [255, 0])	
				pass
			if Particles[i][0] >= systemXBounds[1]:
				Particles[i][3] = -Particles[i][3]
				Particles[i][0] = systemXBounds[1]
			if Particles[i][0]<= systemXBounds[0]: 
				Particles[i][3] = -Particles[i][3]
				Particles[i][0] = systemXBounds[0]

			if Particles[i][1] >= systemYBounds[1]:
				Particles[i][4]  = -Particles[i][4]
				Particles[i][1] = systemYBounds[1]
			if Particles[i][1] <= systemYBounds[0]: 
				Particles[i][4]  = -Particles[i][4] 
				Particles[i][1] = systemYBounds[0]

			if Particles[i][2] >= systemZBounds[1]:
				Particles[i][5] = -Particles[i][5]
				Particles[i][2] = systemZBounds[1]
			if Particles[i][2]<= systemZBounds[0]: 
				Particles[i][5] = -Particles[i][5]
				Particles[i][2] = systemZBounds[0]

			Particles[i][0] += Particles[i][3]
			Particles[i][1] += Particles[i][4] 
			Particles[i][2] += Particles[i][5]
		logging.info("Update Thread finished From Index: " + str(startIndex))
		endEvent.set()

def apply_gravity():
	while True:
		gravityStartEvent.wait()
		gravityStartEvent.clear()
		logging.info("Gravity Thread started")
		for particle in Particles:
			particle[0] =  particle[0] + 10
		logging.info("Gravity Thread finished")
		gravityEndEvent.set()

def render(startIndex, endIndex, startEvent, endEvent):
	while True:
		startEvent.wait()
		logging.info("Raycast Thread started From Index: " + str(startIndex))
		startEvent.clear()
		for i in range(startIndex, endIndex):
			image[Rays[i][3], Rays[i][4], 0] = 0
			image[Rays[i][3], Rays[i][4], 1] = 0
			image[Rays[i][3], Rays[i][4], 2] = 0
			for obj in Particles:
				oc = [cameraPos[0] - obj[0], cameraPos[1] - obj[1], cameraPos[2] - obj[2]]
				if (2 * (oc[0] * Rays[i][0] + oc[1] * Rays[i][1] + oc[2] * Rays[i][2]))**2 - 4 * (Rays[i][0]**2 + Rays[i][1]**2 + Rays[i][2]**2) * ((oc[0]**2 + oc[1]**2 + oc[2]**2) - 5) >= 0:
					image[Rays[i][3], Rays[i][4], 0] = obj[6][0]
					image[Rays[i][3], Rays[i][4], 1] = obj[6][1]
					image[Rays[i][3], Rays[i][4], 2] = obj[6][2]
			pass
		logging.info("Raycast Thread finished From Index: " + str(startIndex))
		endEvent.set()

def displayCallback():
	glutStartEvent.wait()
	glutStartEvent.clear()
	logging.info("GlutLoop Started")
	gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
	gl.glLoadIdentity()
	gl.glRasterPos2i(-1,  -1)
	gl.glPixelZoom(canvasWidth/screenWidth, canvasHeight/screenHeight)
	gl.glDrawPixels(screenWidth, screenHeight, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image)
	gl.glRasterPos2f(-0.9, 0.9)
	glut.glutSwapBuffers()
	glutStartEvent.clear()
	logging.info("GlutLoop Finnished")
	glutEndEvent.set()

def reshapeCallback(width, height):
	gl.glClearColor(1, 1, 1, 1)
	gl.glViewport(0, 0, width, height)
	gl.glMatrixMode(gl.GL_PROJECTION)
	gl.glLoadIdentity()
	gl.glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)	
	gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

def exit():
	for event in updateThreadStartEvents:
		event.set()
	for event in raycastThreadStartEvents:
		event.set()

	gravityStartEvent.set()
	gravityThread.join()

	for thread in updateThreads:
		thread.join()
	for thread in raycastThreads:
		thread.join()
	sys.exit()

def keyboardCallback(key, x, y):
	global ColorType, gravity
	if key == b'\033':
		exit()
	elif key == b'q':
		exit()
	elif key == b'g':
		logging.info("Toggled Gravity")
		gravity = not gravity
	elif key == b'1':
		logging.info("Set Color Type To Block")
		NewColors()
		ColorType = 1
	elif key == b'2':
		logging.info("Set Color Type To Velocity")
		ColorType = 2
	elif key == b'3':
		logging.info("Set Color Type To Total Force")
		ColorType = 3
	elif key == b'4':
		logging.info("Set Color Type To Center Proximity")
		ColorType = 4

def initGlut():
	glut.glutInit()
	glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
	glut.glutInitWindowSize(canvasWidth, canvasHeight)
	glut.glutInitWindowPosition(100, 100)
	glut.glutCreateWindow('Particles')
	glut.glutDisplayFunc(displayCallback)
	glut.glutIdleFunc(displayCallback)
	glut.glutReshapeFunc(reshapeCallback)
	glut.glutKeyboardFunc(keyboardCallback)
	logging.info("Finished Setting Up Glut Loop")
	glut.glutMainLoop()



if __name__ == "__main__":
	logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
	
	generateParticles()
	generateRays()

	glutThread = threading.Thread(target=initGlut)
	glutThread.start()

	gravityThread = threading.Thread(target=apply_gravity)
	gravityThread.start()

	while True:
		if gravity:
			gravityStartEvent.set()
			gravityEndEvent.wait()
			gravityEndEvent.clear()

		for event in updateThreadStartEvents:
			event.set()
		for event in updateThreadEndEvents:
			event.wait()
			event.clear()

		for event in raycastThreadStartEvents:
			event.set()
		for event in raycastThreadEndEvents:
			event.wait()
			event.clear()

		glutStartEvent.set()
		glutEndEvent.wait()
		glutEndEvent.clear()