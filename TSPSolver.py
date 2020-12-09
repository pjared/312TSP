#!/usr/bin/python3

from typing import DefaultDict
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
import math
import random
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

	def greedy(self, time_allowance=60.0):
		bssf = None
		cities = self._scenario.getCities()
		self.numCities = len(cities)
		solutionList = {}
		startTime = time.time()

		# Respect the time limit!
		while time.time() - startTime < time_allowance:
			# By using this indexLoop, the time required is increased, but
			# it also means that we can find the lowest time from any starting point
			# Time: O(n^2 log^2 n)
			for indexLoop in range(len(cities)):
				startCity = cities[indexLoop]
				tourPath = []
				tourPath.append(startCity)

				# Deep copy to avoid modifying the original list of cities
				# (We'll need it later)
				remainingCities = copy.deepcopy(cities)
				currentCity = startCity
				del remainingCities[indexLoop]

				# We'll be deleting the cities as we go, so once it's become
				# zero, we can stop the loop.
				# Time: O(n log^2 n)
				while len(remainingCities):
					cityCosts = self.getClosestCities(currentCity, remainingCities)

					# Find the closest city (they're sorted, so it's the first item in the array)
					closestCityInfo = cityCosts[0]
					closestCityLoc = remainingCities.index(closestCityInfo[0])
					closestCity = remainingCities[closestCityLoc]

					# Make absolutely certain the edge ACTUALLY exists
					if not self._scenario._edge_exists[currentCity._index][closestCity._index]:
						break
					del remainingCities[closestCityLoc]
					tourPath.append(closestCity)

					# Start the next iteration from this city
					currentCity = closestCity

				# If, for whatever reason, the tour is made impossible,
				# continue on to the next starting point.
				if (len(remainingCities)):
					continue

				# Once we've made a full tour, we create the results item
				else:
					bssf = TSPSolution(tourPath)
					endTime = time.time()
					results = {}
					results['cost'] = bssf.cost
					results['time'] = endTime - startTime
					results['count'] = None
					results['soln'] = bssf
					results['max'] = None
					results['total'] = None
					results['pruned'] = None

					# Put the results into the full list of solutions.
					# We can find the lowest cost later.
					solutionList[indexLoop] = results
					continue

			# Search for the  lowest cost among the solution list.
			self.lowest_cost = float("inf")
			for key, solution in solutionList.items():
				if solution['cost'] < self.lowest_cost:
					self.lowest_cost = solution['cost']
					lowest = solution

			# Return the lowest cost from among the found solutions.
			return lowest

	# Finds the distances between the current city and all of the other ones.
	# Then sorts the array so we can quickly access the closest city.
	# remainingCities allows us to exclude cities already passed in this tour
	#
	# Time Complexity: O(n log n), which is Python's sorted() command's time complexity
	# Space Complexity: O(n), as it must store the array before returning it
	def getClosestCities(self, city, remainingCities):
		costs = {}
		for destCity in remainingCities:
			costs[destCity] = city.costTo(destCity)
		sortedCosts = sorted(costs.items(), key=lambda item: item[1])
		return sortedCosts


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
					for j in range(len(self.priorityQueue)):
						if childNode.getBound() > self.priorityQueue[j].getBound():
							if len(path) < len(self.priorityQueue[j].getPath()):
								self.priorityQueue.insert(j, childNode) #less depth, higher bound
								inserted = True
					if len(listChildNodes) == 0 or not inserted:
						self.priorityQueue.append(childNode)
				else:
					self.totalPrunes += 1
			else:
				self.totalPrunes += 1

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

	# Space complexity: O(n)
	# Time complexity: greedy algorithm + while loop * (generateNextState() + probabilityTest())
	#                       ?           +    O(1)    * (       ?             +     O(1)        )
	def fancy(self, time_allowance=60.0):
		results = {}
		count = 0
		T = 30  # Starting temp
		alpha = .99  # factor of Temperature's cooling

		# time/space complexity = greedy algorithm
		currentState = self.getInitialState()
		self.bssf = currentState
		temp = T
		startTime = time.time()
		tempThreshold = .1  # The loop ends when it cools to this temp (or time runs out)

		# Space complexity: At most stores 3 full states (bssf, current, and next). O(3*n) = O(n)
		while temp > tempThreshold and (time.time() - startTime) < time_allowance:
			temp = temp * alpha

			if currentState.cost != float("inf"):
				count += 1

			nextState = self.generateNextState()

			improvedCost = currentState.cost - nextState.cost  # I switched these since we're looking for a min not a max.
			# That makes sense mathematically, right?

			if nextState.cost > self.bssf.cost:
				self.bssf = nextState


			if improvedCost > 0:
				currentState = nextState
			elif self.probabilityTest():  # time/space O(1)
				currentState = nextState


		#annealing
		#500 random solutions
		#comparisons
		#takes out solutions
		endTime = time.time()

		results['cost'] = self.bssf.cost
		results['time'] = endTime - startTime
		results['count'] = count
		results['soln'] = self.bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		Returns random permutation of cities (or could use default algorithm)
		</summary>
		<returns> TSP Solution </returns>
	'''

	def getInitialState(self):
		# return self.defaultRandomTour()
		return self.greedy()

	''' <summary>
		Swaps two random cities. When temp is high, swap can occur with any two cities. As temperature decreases, 
		swaps occur between states that are closer and closer together. 
		</summary>
		<returns> TSP Solution </returns>
	'''
	def generateNextState(self):
		pass

	''' <summary>
		checks to see if:
				e^(improvedCost/temp) > random number between (0,1)
		</summary>
		<returns> Boolean </returns>
	'''

	# Checks whether the algorithm will change its state despite the
	# higher cost. This prevents the algorithm from getting stuck in
	# a local minimum.
	#
	# Returns True when the value of e^(improvedCost/temp) is greater
	# than the random number, meaning that the algorithm should change states.
	# Returns False otherwise, meaning that the algorithm should NOT
	# change states.
	#
	# Note that as the improvedCost becomes more negative, the algorithm
	# is less likely to change the state.
	# Also, as the "temperature" decreases, the algorithm is less likely
	# to change its state.
	#
	# Time Complexity: O(1), as it simply takes two variables and does some math.
	# Space Complexity: O(1), as it stores only two variables.
	def probabilityTest(improvedCost, temp):
		checkedValue = math.exp(improvedCost / temp)
		randomValue = random.random()
		print(checkedValue, " > ", randomValue, " ?")
		if checkedValue > randomValue:
			return True
		else:
			return False





