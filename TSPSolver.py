#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import matrixnode
from matrixnode import matrixnode
import copy



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario
		self.bssf = math.inf
		self.priorityQueue = []
		self.totalPrunes = 0
		self.maxQueSize = 0
		self.totalStates = 0
		self.cities = []
		# 12


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		self.bssf = bssf
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		pass


	def reduceCost(self, matrix, avoidRows, avoidCols):
		totalCost = 0
		#loop through the matrix
		for i in range(len(matrix)):
			if i in avoidRows:
				continue
			hasZero = False
			temp = []
			# Check if it has a 0, store values
			for j in range(len(matrix)):
				temp.append(matrix[i][j])
				if matrix[i][j] == 0:
					hasZero = True
			#get minVal from stored vals, reduce row
			if not hasZero:
				minVal = min(temp)
				for j in range(len(matrix)):
					if matrix[i][j] != math.inf:
						matrix[i][j] -= minVal
				totalCost += minVal

		for i in range(len(matrix)):
			if i in avoidCols:
				continue
			hasZero = False
			temp = []
			for j in range(len(matrix)):
				temp.append(matrix[j][i])
				if matrix[j][i] == 0:
					hasZero = True
			if not hasZero:
				minVal = min(temp)
				for j in range(len(matrix)):
					if matrix[j][i] != math.inf:
						matrix[j][i] -= minVal
				totalCost += minVal
		return totalCost, matrix

	def makeInfinities(self, row, col, matrix):
		# Creating the inf row and col
		for i in range(0, len(matrix)):
			matrix[row][i] = math.inf
			matrix[i][col] = math.inf
		#creating the single point inf
		matrix[col][row] = math.inf
		return matrix

	def getAvoids(self, path):
		avoidRows = []
		avoidCols = []
		#Avoiding the rows and cols that are all inf.
		for i in range(len(path) - 1):
			avoidRows.append(path[i])
		for i in range(1, len(path)):
			avoidCols.append(path[i])
		return avoidRows, avoidCols

	def insertIntoQueue(self, parentNode, curPos):
		#TODO: consider both bound and tree depth(need to add tree depth)
		parentMatrix = parentNode.getMatrix()
		listChildNodes = []# Need to populate this with the arrays that actually have a path

		for i in range(len(parentMatrix)):
			#add the state
			self.totalStates += 1
			temp = parentMatrix[curPos][i]
			if parentMatrix[curPos][i] != math.inf:
				path = copy.deepcopy(parentNode.getPath())
				if i == path[0] and len(path) < len(parentMatrix):
					continue #avoid traveling back before starting
				#infinity out
				childMatrix = self.makeInfinities(curPos, i, copy.deepcopy(parentMatrix))
				path.append(i)
				#get the cols and rows to avoid for reduced cost matrix
				avoidRows, avoidCols = self.getAvoids(path)
				cost, childMatrix = self.reduceCost(childMatrix, avoidRows, avoidCols)
				#get the total cost of the parent bound, the cost to the next point, and reduced cost matrix
				cost = cost + parentNode.getBound() + parentMatrix[curPos][i]
				if self.bssf == None or self.bssf > cost:
					childNode = (matrixnode(childMatrix, cost, path))
					inserted = False
					#Sort insert into the list
					for j in range(len(listChildNodes)): #for 8 change to prio queue, and account for depth
						if childNode.getBound() > listChildNodes[j].getBound():
							listChildNodes.insert(j, childNode)
							inserted = True
					if len(listChildNodes) == 0 or not inserted:
						listChildNodes.append(childNode)
				else:
					self.totalPrunes += 1
			else:
				self.totalPrunes += 1
		self.priorityQueue += listChildNodes

	def checkQueueSize(self):
		#Checking the max queue size every iteration
		if len(self.priorityQueue) > self.maxQueSize:
			self.maxQueSize = len(self.priorityQueue)

	def branchAndBound( self, time_allowance=60.0 ):
		results = self.defaultRandomTour()
		self.cities = self._scenario.getCities()
		ncities = len(self.cities)
		foundTour = False
		count = 1
		self.bssf = results['cost']
		start_time = time.time()
		#initialize the matrix with costs
		initialMatrix = [[0 for x in range(ncities)] for y in range(ncities)]
		for i in range(len(self.cities)):
			for j in range(len(self.cities)):
				initialMatrix[i][j] = self.cities[i].costTo(self.cities[j])
		#reduce the initial matrix
		reduceCost, initialMatrix = self.reduceCost(initialMatrix,[],[])
		self.priorityQueue.append(matrixnode(initialMatrix, reduceCost, [0]))
		bestPath = None
		#start the algorithm
		while len(self.priorityQueue) > 0:
			self.checkQueueSize()
			node = self.priorityQueue.pop()
			path = node.getPath()
			#see if the bound is greater than the bssf
			if node.getBound() > self.bssf:
				self.totalPrunes += 1
				continue
			#check to see if we have crossed the time allowance
			if time.time() - start_time > time_allowance:
				break
			if len(path) == ncities:
				#only update bssf if path is good.
				if node.getBound() < self.bssf:
					count += 1
					foundTour = True
					self.bssf = node.getBound()
					bestPath = node.getPath()
			else:
				self.insertIntoQueue(node, path[len(path) - 1])
		#Out of the loop, so get the actual path
		if foundTour:
			finalPath = []
			for i in range(len(bestPath)):
				finalPath.append(self.cities[bestPath[i]])
			'''for i in range(len(bestPath) - 1):
				if initialMatrix[bestPath[i]][bestPath[i + 1]]  == math.inf:
					print("oops")'''
			self.bssf = TSPSolution(finalPath)
		end_time = time.time()
		results['cost'] = self.bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = self.bssf if foundTour else None
		results['max'] = self.maxQueSize
		results['total'] = self.totalStates
		results['pruned'] = self.totalPrunes
		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
	def fancy( self,time_allowance=60.0):
		#annealing
		#500 random solutions
		#comparisons
		#takes out solutions
		
		pass
		



