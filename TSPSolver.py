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



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario
		self.bssf = None
		self.priorityQueue = None


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
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	def reduceCost(self, matrix):
		#TODO: Need to make sure that this works
		totalCost = 0
		for row in matrix:
			hasZero = False
			for element in row:
				if element == 0:
					hasZero = True
			if not hasZero:
				minVal = min(row)
				for element in row:
					if element != math.inf:
						element -= minVal
				totalCost += minVal
		for column in matrix:
			hasZero = False
			for element in column:
				if element == 0:
					hasZero = True
			if not hasZero:
				minVal = min(column)
				for element in column:
					if element != math.inf:
						element -= minVal
				totalCost += minVal

	def makeInfinities(self, row, col, matrix):
		#TODO: Need to make sure that this works
		for i in range(0, len(matrix)):
			matrix[row][i] = math.inf
		for i in range(0, len(matrix)):
			matrix[i][col] = math.inf

	def checkValidPath(self):
		#for this I should make sure that the length of the list is the some preset length of all the cities
		#If i'm going to go this route, I need to make another array that keeps track of the priority queue and its cities,
		#And I also need to make a curVisit list that will have all the cities from the popped array
		#OR I can just make some kind of checking function (probably the easier route)
		#TODO:Implement some sort of fast checker function(keep the other one as backup)
		pass

	def insertIntoQueue(self, parentMatrix, curPos):
		#Can probably just pass in the base array and then work from there
		#Prune the infinities ONLY
		#only inserting the valid states into queue
		#TODO: need to call make infinity function on this before I send it in here
		listCityPath = []
		listArrays = []  # Need to populate this with the arrays that actually have a path
		for i in range(len(parentMatrix)):
			if parentMatrix[curPos][i] == math.inf:
				listCityPath.append(i)
			else:
				#TODO: PRUNE HERE
				totalPrunes = 0
				pass
		for i in range(len(listCityPath)):
			#Now we're going to pass it into infinity
			#Then append it to listArrays 
			#TODO: Make sure that original matrix is not overidden after passed into inifity
			pass
		#Then we reduce cost these matrixes
		#Then we add them to the queue

		'''
		listPos = []
		for i in range(0, len(listArrays)):
			listPos.append(self.reduceCost(listArrays[i]))
		queuedArrays = []
		while len(listArrays) > 0:
			biggestVal = listPos[0]
			for i in range(0,len(listPos)):
				if listPos(i) > biggestVal:
					biggestVal = listPos(i)
					#largest, so need to add
					queuedArrays.append(listArrays[i])
					#TODO: Find out how to pop the i'th value for prio queue
					listArrays.remove(i)
		#TODO: Make sure this works (is it even necessary?)
		#TODO: smallest cost array should be on top so we can directly add this to priority queue
		return queuedArrays
		'''

	def branchRecursion(self, time_allowance, matrix):
		#TODO:Write this function and should almost be done
		#probably need to follow the slides for this one
		pass

	def branchAndBound( self, time_allowance=60.0 ):
		#TODO:See what 'cities' returns in default
		#TODO:Make a double matrix with the length
		#TODO:Write the recursive function
		#This function is just all setup for the recursive function.
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		if time.time() - start_time < time_allowance:
			return "Timeout"
		#steal initial from random?

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
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
		



