import pyspark
from pyspark import SparkContext
import math
from itertools import combinations
import time 
import sys
import csv
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

start = time.time()
trainFile = sys.argv[1]
testFile = sys.argv[2]
case = int(sys.argv[3])
outFile = sys.argv[4]
sc = SparkContext("local[*]", "LSH")
# trainData = sc.textFile("C:\\Users\\cpiyu\\Downloads\\INF 553 Data Mining\\Assignments\\Assignment 3\\yelp_train.csv")
trainData = sc.textFile(trainFile)
trainDataRDD = trainData.map(lambda x:x.split(",")).filter(lambda x:str(x[0]) != 'user_id')

# validationData = sc.textFile("C:\\Users\\cpiyu\\Downloads\\INF 553 Data Mining\\Assignments\\Assignment 3\\yelp_val.csv")
validationData = sc.textFile(testFile)
validationDataRDD = validationData.map(lambda x:x.split(",")).filter(lambda x:str(x[0]) != 'user_id').map(lambda x: ((x[0],x[1])))
tempValidationRDD = validationData.map(lambda x:x.split(",")).filter(lambda x:str(x[0]) != 'user_id').map(lambda x: ((x[0],x[1]),x[2]))
userToBusinessValue = trainDataRDD.map(lambda x:(x[0],[(x[1],float(x[2]))])).reduceByKey(lambda x,y:x+y).collectAsMap()
#(User,(Business,Rating))
businessToUserValue = trainDataRDD.map(lambda x:(x[1],[(x[0],float(x[2]))])).reduceByKey(lambda x,y:x+y).collectAsMap()

if(case == 1):

	testTrainCombinedRDD = sc.union([trainDataRDD,validationDataRDD])
	
	def creatSet(rddtoList):
		returnSet = set()
		for val in rddtoList:
			returnSet.add(val)
		return returnSet

	users = set()
	businesses = set()
	userToInt = dict()
	businessToInt = dict()

	userRDD = testTrainCombinedRDD.map(lambda x:x[0])
	businessRDD = testTrainCombinedRDD.map(lambda x:x[1])

	users = creatSet(userRDD.collect())
	businesses = creatSet(businessRDD.collect())
	intToUser = dict()
	for i,val in enumerate(users):
		userToInt[val] = i
		intToUser[i] = val
		# print(i, val)
	# print(intToUser)
	# print("users : ",len(userToInt))
	intToBusiness = dict()
	for i,val in enumerate(businesses):
		businessToInt[val] = i
		intToBusiness[i] = val
	# print("businesses : ",len(businessToInt))
	ratings = trainDataRDD.map(lambda x:Rating(int(userToInt[x[0]]),int(businessToInt[x[1]]),float(x[2])))
	# print(ratings.collect()[0:5])
	# print(a.collect()[0:3])
	rank = 5
	numIterations = 10

	# validationDataRDD = testWithoutHeader.filter(lambda x:x[0] in userToInt.keys()).filter(lambda x:x[1] in businessToInt.keys()).map(lambda x:((userToInt[x[0]],businessToInt[x[1]])))
	testingData = validationDataRDD.map(lambda x:(userToInt[x[0]],businessToInt[x[1]]))
	val_rdd = testingData.map(lambda x : ((x[0], x[1]), 0))
	model = ALS.train(ratings, rank, numIterations,lambda_=0.1).predictAll(testingData).map(lambda x:((x[0],x[1]),x[2]))
	missingValuesPrediction = val_rdd.subtractByKey(model).map(lambda x: (x[0], 3))
	# print("Missing : ",missingValuesPrediction.take(), model.count())
	allPredictions = sc.union([model,missingValuesPrediction])
	# print (allPredictions.take(1000))
	print(val_rdd.take(1000))
	print (len(intToUser), len(intToBusiness), allPredictions.count())
	# print (intToBusiness)
	a = allPredictions.map(lambda x:( (intToUser[x[0][0]], intToBusiness[x[0][1]]),x[1]))
	# print("Pred : ",a.take(5))
	# print("Val : ",tempValidationRDD.take(2))
	
	# print(ratesAndPreds.take(2))



	print(a.count())
	# model_1 = ratesAndPreds.map(lambda x:((intToUser[x[0][0]],intToBusiness[x[0][1]]),x[1])).collectAsMap()
	final_answer = a.collectAsMap()
	print(len(final_answer))
	with open(outFile,'w',newline='') as outTask1:
		csv_out = csv.writer(outTask1)
		csv_out.writerow(['user_id',' business_id',' prediction'])
		for pair,pred in final_answer.items():
			csv_out.writerow([pair[0],pair[1],pred])


	# ratesAndPreds = tempValidationRDD.join(a)
	# # ratesAndPreds = validationDataRDD.map(lambda r: ((userToInt[r[0]],businessToInt[r[1]]), r[2])).join(model)
	# MSE = ratesAndPreds.map(lambda r: ( round(float(r[1][0]), 1) - r[1][1])**2).mean()
	# RMSE = math.sqrt(MSE)
	# print("Root Mean Squared Error = " + str(RMSE))
	# print('Predicted')
	# print(time.time() - start)
	# sc.stop()

elif(case == 2):
	def pearsonCorrelation(activeUserDict,otherUserDict,intersectionBusiness,activeUserAverage,averageOfOtherUser):
		activeUserSum = 0
		otherUserSum = 0
		activeUserAverage = 0
		otherUserAverage = 0
		numerator = 0
		firstValDenominator = 0
		secondValDenominator = 0
		finalDenominator = 0
		# for business in intersectionBusiness:
		# 	activeUserSum += activeUserDict[business]
		# 	otherUserSum  += otherUserDict[business]
		# activeUserAverage = activeUserSum/len(intersectionBusiness)
		# otherUserAverage = otherUserSum/len(intersectionBusiness)

		for business in intersectionBusiness:
			numerator += (activeUserDict[business] - activeUserAverage) * (otherUserDict[business] - averageOfOtherUser)
			firstValDenominator += (activeUserDict[business] - activeUserAverage) ** 2
			secondValDenominator += (otherUserDict[business] - averageOfOtherUser) ** 2
		finalDenominator = math.sqrt(firstValDenominator) * math.sqrt(secondValDenominator)
		if(finalDenominator != 0):
			return numerator/finalDenominator
		else: return 0
			# return numerator/finalDenominator
		# else: return 0

	def calculateSimilarity(pearsonCorrelationBusiness,avgOfRatings,otherUserAvgDict,allUserRatingDict,testBusiness):
		similarity = 0
		denominator = 0
		for user in pearsonCorrelationBusiness:
			# print("sbc : ",pearsonCorrelationBusiness[user])
			similarity += (allUserRatingDict[user] - otherUserAvgDict[user]) * pearsonCorrelationBusiness[user]
			denominator += abs(pearsonCorrelationBusiness[user])
		if(denominator == 0):
			return avgOfRatings
		else:
			return avgOfRatings + similarity/denominator

	def getSimilarUsers(x):
		activeUser = x[0]
		testBusiness = x[1]
		userAndRating = list()
		userSet = set()
		intersectionBusiness = set()
		similarity = 0
		otherUserBusinesses = set()
		ratingList = list()
		pearsonCorrelationBusiness = dict()
		otherUserDict = dict()
		otherUserAvgDict = dict()

		#Active User Info
		# print("Test User : ",activeUser)
		# print("Test Business",testBusiness)
		activeUserRating = userToBusinessValue.get(activeUser)
		# print("a : ",activeUserRating)
		activeUserBusinesses = set()
		activeUserDict = dict()
		sumOfRatings = 0
		avgOfRatings = 0
		allUserRatingDict = dict()
		for val in activeUserRating:
			sumOfRatings += val[1]
			activeUserBusinesses.add(val[0])
			activeUserDict[val[0]] = val[1]
		avgOfRatings = sumOfRatings/len(activeUserRating)
		# print("len : ",len(activeUserDict.keys()))
		# print("a : ",sorted(activeUserDict.keys()))
		if(activeUser not in userToBusinessValue.keys()):
			allUsersAndRatings = list()
			totalRating = 0
			averageRating = 0
			allUsersAndRatings = businessToUserValue[testBusiness]
			for i in allUsersAndRatings[1]:
				totalRating += i
			averageRating = round((totalRating/len(allUsersAndRatings)),2)
			return (x,averageRating)
		if(testBusiness in businessToUserValue.keys()):
			userAndRating = businessToUserValue[testBusiness]
			# print("a : ",userAndRating)
			for user in userAndRating:
				userSet.add((activeUser,user[0]))
				# print(user[0],user[1])
				allUserRatingDict[user[0]] = float(user[1])

			for user in userSet:
				sumOfOtherUserRatings = 0
				averageOfOtherUser = 0
				otherUserDict = dict()
				otherUserBusinesses = set()
				ratingList = userToBusinessValue.get(user[1])
				for rating in ratingList:
					otherUserBusinesses.add(rating[0])
					otherUserDict[rating[0]] = rating[1]
					sumOfOtherUserRatings += rating[1]
				averageOfOtherUser = sumOfOtherUserRatings/len(otherUserBusinesses)
				otherUserAvgDict[user[1]] = averageOfOtherUser
				# print("len : ",user[1],len(otherUserDict.keys()))
				intersectionBusiness = activeUserBusinesses.intersection(otherUserBusinesses)
				# print("intersection length : ",user,len(intersectionBusiness))
			# print(otherUserDict)
				if(len(intersectionBusiness) != 0):
					weight = pearsonCorrelation(activeUserDict,otherUserDict,intersectionBusiness,avgOfRatings,averageOfOtherUser)
					# print("weight : ",weight)
					pearsonCorrelationBusiness[user[1]] = weight
					# print("pc :",pearsonCorrelationBusiness)
					# otherUserAvgDict[user[1]] = weight[1]
			pearsonFilteredDict = dict()
			for val in pearsonCorrelationBusiness.keys():
				if(pearsonCorrelationBusiness[val] > 0):
					pearsonFilteredDict[val] = pearsonCorrelationBusiness[val]
			similarity = calculateSimilarity(pearsonFilteredDict,avgOfRatings,otherUserAvgDict,allUserRatingDict,testBusiness)
			if(similarity > 5):
				return (x,5)
			elif(similarity < 1): 
				return (x,1)
			else:
				return (x,round(similarity,2))
		else:
			return (x,round(avgOfRatings,2))

	prediction = validationDataRDD.map(lambda x:getSimilarUsers(x)).collectAsMap()
	with open(outFile, 'w', newline='') as out:
		csv_out=csv.writer(out)
		csv_out.writerow(['user_id',' business_id',' prediction'])
		for pair,itemset in prediction.items():
			csv_out.writerow([pair[0],pair[1],itemset])
	# print(prediction.take(1))
	# validationGivenRatings = validationData.map(lambda x:x.split(",")).filter(lambda x:str(x[0]) != 'user_id').map(lambda x:((x[0],x[1]),x[2]))
	# allValidationStrings = prediction.join(validationGivenRatings)
	# MSE = allValidationStrings.map(lambda r: (float(r[1][0]) - float(r[1][1]))**2).mean()
	# RMSE = math.sqrt(MSE)
	# print("Root Mean Squared Error = " + str(RMSE))
	
	sc.stop()

elif(case == 3):
	def pearsonCorrelation(activeUserRatingdict,otherUserRatings,coratedUsers,averageofActiveBusiness,otherBusinessAverage):
	    # firstUserRating = 0
	    # secondUserRating = 0
	    # firstUserAverage = 0
	    # secondUserAverage = 0
	    numerator = 0
	    firstDenominatorValue = 0
	    secondDenominatorValue = 0
	    finalDenominator = 0

	    # for user in coratedUsers:
	    #     firstUserRating += activeUserRatingdict[user]
	    #     secondUserRating += otherUserRatings[user]
	    # firstUserAverage = firstUserRating/len(coratedUsers)
	    # secondUserAverage = secondUserRating/len(coratedUsers)
	    # print(user,firstUserAverage,secondUserAverage)
	    for user in coratedUsers:
	        numerator += (activeUserRatingdict[user] - averageofActiveBusiness) * (otherUserRatings[user] - otherBusinessAverage)
	        firstDenominatorValue += (activeUserRatingdict[user] - averageofActiveBusiness) ** 2
	        secondDenominatorValue += (otherUserRatings[user] - otherBusinessAverage) ** 2
	    finalDenominator = math.sqrt(firstDenominatorValue) * math.sqrt(secondDenominatorValue)
	    if(finalDenominator != 0):
	        return (numerator/finalDenominator)
	    else:
	        return 0 
	def calculatePrediction(allBusinessRatingDict,pearsonCorrelationBusiness,otherBusinessAverageDict,averageofActiveBusiness):
	    numerator = 0
	    denominator = 0
	    for business in pearsonCorrelationBusiness.keys():
	        numerator += (allBusinessRatingDict[business] - otherBusinessAverageDict[business])*pearsonCorrelationBusiness[business]
	        denominator += abs(pearsonCorrelationBusiness[business])
	    if(denominator != 0):
	        return averageofActiveBusiness + numerator/denominator
	    else: return averageofActiveBusiness

	def getSimilarBusiness(x):
	    activeBusiness = x[1]
	    testUser = x[0]
	    businessSet = set()
	    activeBusinessUsers = dict()
	    otherBusinessDict = dict()
	    allBusinessRatingDict = dict()
	    userAndRatings = list()
	    activeUserRatingdict = dict()
	    coratedUsers = set()
	    pearsonCorrelationBusiness = dict()
	    #Get list of businesses rated by active user

	    if testUser not in userToBusinessValue.keys() and activeBusiness not in businessToUserValue.keys():
	        return (x, 3.0)
	    if testUser not in userToBusinessValue.keys():
	        userAndRatingsActive = businessToUserValue[activeBusiness]
	        sumOfActiveBusiness = 0
	        averageofActiveBusiness = 0
	        for user in userAndRatingsActive:
	            sumOfActiveBusiness += user[1]
	        averageofActiveBusiness = sumOfActiveBusiness/len(userAndRatingsActive)
	        return (x, averageofActiveBusiness)
	    getAllTestBusinesses = userToBusinessValue[testUser]

	    #Calculate average of active business

	    #Form (active business,business) pairs
	    sumOfBusiness = 0
	    activeUserAverage = 0
	    for testBusiness in getAllTestBusinesses:
	        sumOfBusiness += testBusiness[1]
	        businessSet.add((activeBusiness,testBusiness[0]))
	        allBusinessRatingDict[testBusiness[0]] = float(testBusiness[1])
	    activeUserAverage = sumOfBusiness/len(getAllTestBusinesses)
	    # print(activeUserAverage)
	    #BusinessSet --> (ActiveBusiness,SimilarBusiness)
	    #Get all USers who rated active business
	    if(activeBusiness in businessToUserValue.keys()):
	    
	        usersRatingActiveBusiness = set()
	        userAndRatingsActive = list()
	        userAndRatingsActive = businessToUserValue[activeBusiness]
	        # print("userAndRatingsActive : ",userAndRatingsActive)
	        sumOfActiveBusiness = 0
	        averageofActiveBusiness = 0
	        for user in userAndRatingsActive:
	            # print("user : ",user[0])
	            sumOfActiveBusiness += user[1]
	            usersRatingActiveBusiness.add(user[0])
	            activeUserRatingdict[user[0]] = user[1]
	        averageofActiveBusiness = sumOfActiveBusiness/len(userAndRatingsActive)
	        # print("abc : ",usersRatingActiveBusiness)
	        #Get all users who rated other business
	        otherBusinessAverageDict = dict()
	        for business in businessSet:
	            # print("b : ",business)
	            otherUsers = set()
	            otherUserRatings = dict()
	            sumOfBusiness = 0
	            otherBusinessAverage = 0
	            userAndRatings = businessToUserValue.get(business[1])
	            for user in userAndRatings:
	                sumOfBusiness += user[1]
	                otherUsers.add(user[0])
	                otherUserRatings[user[0]] = user[1]
	            otherBusinessAverage = sumOfBusiness/len(userAndRatings)
	            otherBusinessAverageDict[business[1]] = otherBusinessAverage
	            coratedUsers = usersRatingActiveBusiness.intersection(otherUsers)
	            # print("len : ",business,len(coratedUsers))
	            if(len(coratedUsers) != 0):
	                weight = pearsonCorrelation(activeUserRatingdict,otherUserRatings,coratedUsers,averageofActiveBusiness,otherBusinessAverage)
	                # print(weight)
	                pearsonCorrelationBusiness[business[1]] = weight 

	        positivePearsonDict = dict()
	        for pearsonCorrelationKey in pearsonCorrelationBusiness.keys():
	            if(pearsonCorrelationBusiness[pearsonCorrelationKey] > 0):
	                positivePearsonDict[pearsonCorrelationKey] = pearsonCorrelationBusiness[pearsonCorrelationKey]

	        # if(len(positivePearsonDict) > 15):
	        #     a = sorted(positivePearsonDict.items(),key=lambda x:-x[1])
	        #     a = a[0:15]
	        #     pearsonCorrelationBusinessDict = OrderedDict(a)
	        #     prediction = calculatePrediction(allBusinessRatingDict,pearsonCorrelationBusinessDict)
	        # else:
	        prediction = calculatePrediction(allBusinessRatingDict,positivePearsonDict,otherBusinessAverageDict,averageofActiveBusiness)

	        if(prediction > 5):
	            prediction = 5
	        elif(prediction < 1) and (prediction != 0):
	            prediction = 1
	        if (prediction == 0):
	            prediction = averageofActiveBusiness
	        return (x,prediction)
	        
	    else:
	        return (x,activeUserAverage)


	finalPrediction = validationDataRDD.map(lambda x:getSimilarBusiness(x)).collectAsMap()
	with open(outFile, 'w', newline='') as out:
		csv_out=csv.writer(out)
		csv_out.writerow(['user_id',' business_id',' prediction'])
		for pair,itemset in finalPrediction.items():
			csv_out.writerow([pair[0],pair[1],itemset])

	# prediction = validationDataRDD.map(lambda x:getSimilarBusiness(x))
	# validationGivenRatings = validationData.map(lambda x:x.split(",")).filter(lambda x:str(x[0]) != 'user_id').map(lambda x:((x[0],x[1]),x[2]))
	# allValidationStrings = prediction.join(validationGivenRatings)
	# print(allValidationStrings.take(1000))
	# MSE = allValidationStrings.map(lambda r: (float(r[1][0]) - float(r[1][1]))**2).mean()
	# RMSE = math.sqrt(MSE)
	# print("Root Mean Squared Error = " + str(RMSE))

	sc.stop()	