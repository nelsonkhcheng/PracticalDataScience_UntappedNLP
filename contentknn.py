from surprise import AlgoBase
from surprise import PredictionImpossible
import math
import numpy as np
import heapq
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


class ContentKNNAlgorithm(AlgoBase):  

  def __init__(self, k=40, sim_options={}):
    AlgoBase.__init__(self)
    self.k = k
    self.df_features = False
    self.years = { }
    

  def setFeatures(self, features_input):
    self.df_features = True    
    self.years = features_input[["RowID", "Year"]].to_dict()
    self.reviewerReviewCounts = features_input[["RowID", "ReviewerReviewCount"]].to_dict()
    self.beerReviewCounts = features_input[["RowID", "BeerReviewCount"]].to_dict()
    self.abvs = features_input[["RowID", "ABV"]].to_dict()
    self.dayOfWeeks = features_input[["RowID", "DayofWeek"]].to_dict()
    self.dayOfMonths = features_input[["RowID", "DayofMonth"]].to_dict()
    self.months = features_input[["RowID", "Month"]].to_dict()


  # Just using Years, for first tinkering testing run
  def setYears(self, years_input):
    self.df_features = True
    self.years = years_input    

  def fit(self, trainset):

    if self.df_features is None:
      print("Make sure to set the features in the algorithm before trying to fit")
      return

    AlgoBase.fit(self, trainset)
    # Compute item similarity matrix based on content attributes
    # Load up genre vectors for every movie

    print("Computing content-based similarity matrix...")
    # Compute genre distance for every movie combination as a 2x2 matrix
    self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

    reportingCount = 0    
    for thisRating in range(self.trainset.n_items):
      if (thisRating % 1000 == 0):
        print(thisRating, " of ", self.trainset.n_items)

      reportingPerItemCount = 0
      for otherRating in range(thisRating+1, self.trainset.n_items):
        thisMovieID = int(self.trainset.to_raw_iid(thisRating))
        otherMovieID = int(self.trainset.to_raw_iid(otherRating))

        if reportingCount < 5 and reportingPerItemCount < 3:
          print("  Processing thisMovieID: " + str(thisMovieID) + " otherMovieID: " + str(otherMovieID))
        
        # genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)

        reviewerReviewCountSimilarity = self.computeReviewerReviewCountSimilarity(thisMovieID, otherMovieID)
        beerReviewCountSimilarity = self.computeBeerReviewCountSimilarity(thisMovieID, otherMovieID)
        abvSimilarity = self.computeABVSimilarity(thisMovieID, otherMovieID)
        dayofWeekSimilarity = self.computeDayOfWeekSimilarity(thisMovieID, otherMovieID)
        dayOfMonthSimilarity = self.computeDayOMonthSimilarity(thisMovieID, otherMovieID)
        monthSimilarity = self.computeMonthSimilarity(thisMovieID, otherMovieID)
        yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID)
        #mesSimilarity = self.computeMiseEnSceneSimilarity(thisMovieID, otherMovieID, mes)
        
        self.similarities[thisRating, otherRating] = reviewerReviewCountSimilarity * beerReviewCountSimilarity * abvSimilarity * \
                        dayofWeekSimilarity * dayOfMonthSimilarity * monthSimilarity * yearSimilarity
        # self.similarities[thisRating, otherRating] = yearSimilarity

        self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]

        reportingPerItemCount += 1

      reportingCount += 1
    print("...done.")
    return self       

  def computeReviewerReviewCountSimilarity(self, row1Id, row2Id):
    if row1Id in self.years and row2Id in self.years:
      diff = abs(self.reviewerReviewCounts[row1Id] - self.reviewerReviewCounts[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0
      

  def computeBeerReviewCountSimilarity(self, row1Id, row2Id):
    if row1Id in self.beerReviewCounts and row2Id in self.beerReviewCounts:
      diff = abs(self.beerReviewCounts[row1Id] - self.beerReviewCounts[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeABVSimilarity(self, row1Id, row2Id):
    if row1Id in self.abvs and row2Id in self.abvs:
      diff = abs(self.abvs[row1Id] - self.abvs[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeDayOfWeekSimilarity(self, row1Id, row2Id):
    if row1Id in self.dayOfWeeks and row2Id in self.dayOfWeeks:
      diff = abs(self.dayOfWeeks[row1Id] - self.dayOfWeeks[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeDayOMonthSimilarity(self, row1Id, row2Id):
    if row1Id in self.dayOfMonths and row2Id in self.dayOfMonths:
      diff = abs(self.dayOfMonths[row1Id] - self.dayOfMonths[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeMonthSimilarity(self, row1Id, row2Id):
    if row1Id in self.months and row2Id in self.months:
      diff = abs(self.months[row1Id] - self.months[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeYearSimilarity(self, row1Id, row2Id):
    if row1Id in self.years and row2Id in self.years:
      diff = abs(self.years[row1Id] - self.years[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0
      


  def computeNumericalSimilarity(self, row1Id, row2Id, colName):
    if ((self.df_features["RowID"] == row1Id).any() & (self.df_features["RowID"] == row2Id).any()):
      row1ColValue = self.df_features[self.df_features["RowID"] == row1Id][colName].item()
      row2ColValue = self.df_features[self.df_features["RowID"] == row2Id][colName].item()
      diff = abs(row1ColValue - row2ColValue)
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0

  # def computeMiseEnSceneSimilarity(self, movie1, movie2, mes):
  #   mes1 = mes[movie1]
  #   mes2 = mes[movie2]
  #   if (mes1 and mes2):
  #     shotLengthDiff = math.fabs(mes1[0] - mes2[0])
  #     colorVarianceDiff = math.fabs(mes1[1] - mes2[1])
  #     motionDiff = math.fabs(mes1[3] - mes2[3])
  #     lightingDiff = math.fabs(mes1[5] - mes2[5])
  #     numShotsDiff = math.fabs(mes1[6] - mes2[6])
  #     return shotLengthDiff * colorVarianceDiff * motionDiff * lightingDiff * numShotsDiff
  #   else:
  #     return 0

  def estimate(self, u, i):
    if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
      raise PredictionImpossible('User and/or item is unkown.')
    # Build up similarity scores between this item and everything the user rated
    neighbors = []
    for rating in self.trainset.ur[u]:
      similarity = self.similarities[i,rating[0]]
      neighbors.append( (similarity, rating[1]) )
    # Extract the top-K most-similar ratings
    k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
    # Compute average sim score of K neighbors weighted by user ratings
    simTotal = weightedSum = 0
    for (simScore, rating) in k_neighbors:
      if (simScore > 0):
          simTotal += simScore
          weightedSum += simScore * rating
    if (simTotal == 0):
      raise PredictionImpossible('No neighbors')
    predictedRating = weightedSum / simTotal
    return predictedRating





class ContentKNNFullCosSimilarityAlgorithm(AlgoBase):  

  def __init__(self, k=40, sim_options={}):
    AlgoBase.__init__(self)
    self.k = k
    self.df_features = None
    self.years = { }
    

  def setFeatures(self, features_input):
    self.df_features = features_input    

  
  def setYears(self, years_input):
    self.df_features = True
    self.years = years_input    

  def fit(self, trainset):

    if self.df_features is None:
      print("Make sure to set the features in the algorithm before trying to fit")
      return

    AlgoBase.fit(self, trainset)
    # Compute item similarity matrix based on content attributes
    # Load up genre vectors for every movie

    print("Computing content-based similarity matrix...")

    dfFeaturesToUse = self.df_features.drop(columns="RowID")

    # Compute genre distance for every movie combination as a 2x2 matrix
    self.similarities = cosine_similarity(dfFeaturesToUse, dfFeaturesToUse)

    del dfFeaturesToUse

    print("...done.")
    return self       

  def computeReviewerReviewCountSimilarity(self, row1Id, row2Id):
    if row1Id in self.years and row2Id in self.years:
      diff = abs(self.reviewerReviewCounts[row1Id] - self.reviewerReviewCounts[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0
      

  def computeBeerReviewCountSimilarity(self, row1Id, row2Id):
    if row1Id in self.beerReviewCounts and row2Id in self.beerReviewCounts:
      diff = abs(self.beerReviewCounts[row1Id] - self.beerReviewCounts[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeABVSimilarity(self, row1Id, row2Id):
    if row1Id in self.abvs and row2Id in self.abvs:
      diff = abs(self.abvs[row1Id] - self.abvs[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeDayOfWeekSimilarity(self, row1Id, row2Id):
    if row1Id in self.dayOfWeeks and row2Id in self.dayOfWeeks:
      diff = abs(self.dayOfWeeks[row1Id] - self.dayOfWeeks[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeDayOMonthSimilarity(self, row1Id, row2Id):
    if row1Id in self.dayOfMonths and row2Id in self.dayOfMonths:
      diff = abs(self.dayOfMonths[row1Id] - self.dayOfMonths[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeMonthSimilarity(self, row1Id, row2Id):
    if row1Id in self.months and row2Id in self.months:
      diff = abs(self.months[row1Id] - self.months[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0


  def computeYearSimilarity(self, row1Id, row2Id):
    if row1Id in self.years and row2Id in self.years:
      diff = abs(self.years[row1Id] - self.years[row2Id])
      sim = math.exp(-diff / 10.0)
      return sim
    else:
      return 0
      


  # def computeNumericalSimilarity(self, row1Id, row2Id, colName):
  #   if ((self.df_features["RowID"] == row1Id).any() & (self.df_features["RowID"] == row2Id).any()):
  #     row1ColValue = self.df_features[self.df_features["RowID"] == row1Id][colName].item()
  #     row2ColValue = self.df_features[self.df_features["RowID"] == row2Id][colName].item()
  #     diff = abs(row1ColValue - row2ColValue)
  #     sim = math.exp(-diff / 10.0)
  #     return sim
  #   else:
  #     return 0

  # def computeMiseEnSceneSimilarity(self, movie1, movie2, mes):
  #   mes1 = mes[movie1]
  #   mes2 = mes[movie2]
  #   if (mes1 and mes2):
  #     shotLengthDiff = math.fabs(mes1[0] - mes2[0])
  #     colorVarianceDiff = math.fabs(mes1[1] - mes2[1])
  #     motionDiff = math.fabs(mes1[3] - mes2[3])
  #     lightingDiff = math.fabs(mes1[5] - mes2[5])
  #     numShotsDiff = math.fabs(mes1[6] - mes2[6])
  #     return shotLengthDiff * colorVarianceDiff * motionDiff * lightingDiff * numShotsDiff
  #   else:
  #     return 0

  def estimate(self, u, i):
    if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
      raise PredictionImpossible('User and/or item is unkown.')
    # Build up similarity scores between this item and everything the user rated
    neighbors = []
    for rating in self.trainset.ur[u]:
      similarity = self.similarities[i,rating[0]]
      neighbors.append( (similarity, rating[1]) )
    # Extract the top-K most-similar ratings
    k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
    # Compute average sim score of K neighbors weighted by user ratings
    simTotal = weightedSum = 0
    for (simScore, rating) in k_neighbors:
      if (simScore > 0):
          simTotal += simScore
          weightedSum += simScore * rating
    if (simTotal == 0):
      raise PredictionImpossible('No neighbors')
    predictedRating = weightedSum / simTotal
    return predictedRating





##################

  # def computeGenreSimilarity(self, movie1, movie2, genres):
  #   genres1 = genres[movie1]
  #   genres2 = genres[movie2]
  #   sumxx, sumxy, sumyy = 0, 0, 0
  #   for i in range(len(genres1)):
  #     x = genres1[i]
  #     y = genres2[i]
  #     sumxx += x * x
  #     sumyy += y * y
  #     sumxy += x * y
  #   return sumxy/math.sqrt(sumxx*sumyy)

  # def fit_original(self, trainset):
  #   AlgoBase.fit(self, trainset)
  #   # Compute item similarity matrix based on content attributes
  #   # Load up genre vectors for every movie
  #   # ml = MovieLens()
  #   # genres = ml.getGenres()
  #   # years = ml.getYears()
  #   # mes = ml.getMiseEnScene()
  #   # print("Computing content-based similarity matrix...")
  #   # # Compute genre distance for every movie combination as a 2x2 matrix
  #   # self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
  #   # for thisRating in range(self.trainset.n_items):
  #   #   if (thisRating % 100 == 0):
  #   #     print(thisRating, " of ", self.trainset.n_items)
  #   #   for otherRating in range(thisRating+1, self.trainset.n_items):
  #   #     thisMovieID = int(self.trainset.to_raw_iid(thisRating))
  #   #     otherMovieID = int(self.trainset.to_raw_iid(otherRating))
  #   #     genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
  #   #     yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
  #   #     #mesSimilarity = self.computeMiseEnSceneSimilarity(thisMovieID, otherMovieID, mes)
  #   #     self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity
  #   #     self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
  #   print("...done.")
  #   return self    