from nltk.corpus.reader.mte import xpath
import pandas as pd
import numpy as np
from pathlib import Path
import regex as re
import pickle

from datetime import datetime

import fasttext as ft
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

from surprise import dump
from surprise import accuracy
from surprise.model_selection import train_test_split

import lightgbm as lgb

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from utilities import regex_utility as reutil
from utilities import data_basic_utility as databasic

#########################
### START: Non-Text column formatting functions
#########################



# this column has some empty values. fix these by imputing with the mean
def fixNullABV(dfInput):
  meanAbv = np.mean(dfInput["ABV"])
  dfInput["ABV"] = dfInput["ABV"].fillna(meanAbv)
  return dfInput


# For these Train, Test and Vali files/dataframes, reusable formatting of features functions
def formatDayOfWeek(dfInput):
  def formatDayVal(input):
    input = input.lower()
    if input == "mon":
      return 1
    elif input == "tue":
      return 2
    elif input == "wed":
      return 3
    elif input == "thu":
      return 4
    elif input == "fri":
      return 5
    elif input == "sat":
      return 6
    elif input == "sun":
      return 7

  dfInput["DayofWeek"] = dfInput.apply(lambda x: formatDayVal(x["DayofWeek"]), axis=1)
  return dfInput

def formatMonth(dfInput):
  def formatMonthVal(input):
    input = input.lower()
    if input == "jan":
      return 1
    elif input == "feb":
      return 2
    elif input == "mar":
      return 3
    elif input == "apr":
      return 4
    elif input == "may":
      return 5
    elif input == "jun":
      return 6
    elif input == "jul":
      return 7
    elif input == "aug":
      return 8
    elif input == "sep":
      return 9
    elif input == "oct":
      return 10
    elif input == "nov":
      return 11
    elif input == "dec":
      return 12

  dfInput["Month"] = dfInput.apply(lambda x: formatMonthVal(x["Month"]), axis=1)
  return dfInput
  

def formatTimeToSec(dfInput):
  # Helper function that will turn an hh:mm:ss string into a number in seconds
  def formatTimeToSecVal(input):
    inputTokens = input.split(":")
    if len(inputTokens) != 3:
      # invalid time format, return 0
      return 0
    else:
      hours = int(inputTokens[0])
      minutes = int(inputTokens[1])
      seconds = int(inputTokens[2])
      return (hours * 60 * 60) + (minutes * 60) + seconds

  dfInput["TimeOfDay"] = dfInput.apply(lambda x: formatTimeToSecVal(x["TimeOfDay"]), axis=1)
  return dfInput


# Even though it now holds age, with a 0 for unknown, retain the column name Birthday so as not to confuse come colname lists
def convertBirthdayToAge(dfInput):
  # Helper function to filter out rows without a yeear, otherwise do a basic calc on year according to current year
  def convertBirthdayToAgeVal(input):
    if input.lower() == "unknown":
      return 0
    elif "," not in input:
      return 0
    else:
      tokens = input.split(",")
      year = int(tokens[1])
      age = datetime.now().year - year
      if age > 0:
        return age
      else:
        return 0

  dfInput["Birthday"] = dfInput.apply(lambda x: convertBirthdayToAgeVal(x["Birthday"]), axis=1)  
  return dfInput


# Given a particular ratings dataset (not features), will add a column for the number of Reviews the Reviewer has submitted
# The logic being that a Regression will hopefully find that ratings by more active reviewers are more valuable
def addReviewerReviewCount(dfInput):    
  dfInputReviewCounts = dfInput.groupby("ReviewerID", as_index=False).size()
  dfInputReviewCounts.columns=["ReviewerID", "ReviewerReviewCount"]
 
  dfInput = pd.merge(dfInput, dfInputReviewCounts, how="inner", left_on="ReviewerID", right_on="ReviewerID")

  del dfInputReviewCounts
  return dfInput
  

# Given a particular ratings dataset (not features), will add a column for the number of Reviews the Beer has day submitted
# The logic being that a Regression will hopefully find that ratings for beers that have been reviewed many times are more valuable
def addBeerReviewCount(dfInput):    
  dfInputReviewCounts = dfInput.groupby("BeerID", as_index=False).size()
  dfInputReviewCounts.columns=["BeerID", "BeerReviewCount"]
 
  dfInput = pd.merge(dfInput, dfInputReviewCounts, how="inner", left_on="BeerID", right_on="BeerID")

  del dfInputReviewCounts
  return dfInput


def scaleFeatureDataFrame(dfFullFeatures):
  # Scale the data but make sure not to modify the Row ID
  datacols = ["ReviewerReviewCount", "BeerReviewCount", "ABV", "DayofWeek", "DayofMonth", "Month", "Year", "TimeOfDay", "Birthday"]
  columnsToIgnore = dfFullFeatures.columns.drop(datacols)

  dfFullFeaturesIds = dfFullFeatures[columnsToIgnore]
  dfFullFeaturesData = dfFullFeatures[datacols]

  # print(str(len(datacols)))
  # print(datacols)
  # dfFullFeaturesData.head()

  scaler = StandardScaler()
  dfFullFeaturesData = pd.DataFrame(scaler.fit_transform(dfFullFeaturesData), columns=datacols)
  #dfFullFeaturesData.head()

  # join the ids back to the data
  dfFullFeatures = pd.concat([dfFullFeaturesIds.reset_index(), dfFullFeaturesData], axis=1).drop(columns="index")
  #dfFullFeatures.head(10)

  del dfFullFeaturesIds
  del dfFullFeaturesData

  return dfFullFeatures

  

def scaleMinMaxFeatureDataFrame(dfFullFeatures):
  # Scale the data but make sure not to modify the Row ID
  datacols = ["ReviewerReviewCount", "BeerReviewCount", "ABV", "DayofWeek", "DayofMonth", "Month", "Year", "TimeOfDay", "Birthday"]
  columnsToIgnore = dfFullFeatures.columns.drop(datacols)

  dfFullFeaturesIds = dfFullFeatures[columnsToIgnore]
  dfFullFeaturesData = dfFullFeatures[datacols]

  # print(str(len(datacols)))
  # print(datacols)
  # dfFullFeaturesData.head()

  scaler = MinMaxScaler()
  dfFullFeaturesData = pd.DataFrame(scaler.fit_transform(dfFullFeaturesData), columns=datacols)
  #dfFullFeaturesData.head()

  # join the ids back to the data
  dfFullFeatures = pd.concat([dfFullFeaturesIds.reset_index(), dfFullFeaturesData], axis=1).drop(columns="index")
  #dfFullFeatures.head(10)

  del dfFullFeaturesIds
  del dfFullFeaturesData

  return dfFullFeatures

#########################
### END: Non-Text column formatting functions
#########################


#########################
### START: Ensembling Helper Functions
#########################

# Use this to load a run file and add the predictions from that run to the ensemble dataframe
def joinRunToEnsembleFrame(dfEnsemble, subrunDir, runFile): 
  # Load the Run file   
  runFileName = runFile + ".csv"
  dfRun = pd.read_csv(subrunDir + runFileName)

  # rename the prediction column to the file name and drop the Beer ID and Review ID columns
  dfRun = dfRun.rename(columns={'Predict': runFile })
  del dfRun["BeerID"]
  del dfRun["ReviewerID"]  
  
  # Add the prediction column to the ensemble dataframe via join
  dfEnsemble = pd.merge(dfEnsemble, dfRun, on="RowID")

  # remove the run file from memory then return the ensemble frame
  del dfRun
  return dfEnsemble
  

#########################
### END: Ensembling Helper Functions
#########################  


#########################
### START: FastText NLP Helper Functions
#########################

# Given 3 dataframes (across train, validation and test) and a target column,
# compile the contents of that col across all the frames into one list and convert them all into word token strings that
# can be used to build a language model with fast text. Includes Basic Text Preprocessing
def formatTextColForNLP(df1, df2, df3, colName, featuresDataDir, filePrefix, removeTopFrequentTokens = 0, setTopBigrams = 0):
  
  # Combine all three datasets into one dataset that is just row ids and the target column
  df_combined = df1[["RowID", colName]].append(df2[["RowID", colName]]).append(df3[["RowID", colName]])

  # Remove ascii encoding and punctuation
  df_combined[colName] = df_combined.apply(lambda x: str(x[colName]).encode("ascii", "ignore").decode(), axis=1)
  df_combined[colName] = df_combined.apply(lambda x: reutil.str_strip_punctuation(str(x[colName])), axis=1)

  # convert target column into a list of word tokens  
  lstTokens = df_combined.apply(lambda x: str(x[colName]).split(" "), axis=1)
  lstTokens = lstTokens.to_list()

  # Do Text Preprocessing: 
  # remove capitalisation, single letter tokens and stop words
  lstTokens = list(map(lambda x: list(map(lambda y: y.lower(), x)), lstTokens))
  lstTokens = list(map(lambda x: list(filter(lambda y: len(y) >= 2, x)), lstTokens))

  # sort each token list so we can use bsearch
  nltk.download("stopwords")
  stopwordsSorted = sorted(stopwords.words("english"))  
  lstTokens = list(map(lambda x: databasic.filter_by_words_bsearch(x, stopwordsSorted), lstTokens))

  # Create variables for the words and vocab. When we update out tokens lists, we will want to recompile our word list and vocabulary to keep it up to date
  words, vocab = databasic.createWordsAndVocabForTokenLists(lstTokens)

  # Create a term Frequency distribution
  term_fd = nltk.FreqDist(words)

  # remove single occurrence words
  setSingleWords = set(term_fd.hapaxes())
  lstSingleWords = sorted(list(setSingleWords))
  lstTokens = list(map(lambda x: databasic.filter_by_words_bsearch(x, lstSingleWords), lstTokens))
  
  # find the most commong bigrams and join them together to be a single term
  if setTopBigrams > 0:
    bigrams = ngrams(words, n = 2)
    fdbigram = nltk.FreqDist(bigrams)
    mostFreqBigrams = fdbigram.most_common(setTopBigrams)    
    # print(mostFreqBigrams)

    # Convert the bigrams to a list of bigram strings (with spaces)
    rep_patterns = list(map(lambda x: x[0], mostFreqBigrams))
    rep_patterns = list(map(lambda x: x[0] + " " + x[1], rep_patterns))

    # Create another list which is the replacements with the
    replacements = list(map(lambda x: x.replace(" ", "_"), rep_patterns))

    lstTokens = [" ".join(tokens) for tokens in lstTokens] # convert all the token lists back into a single string

    # Loop thought and basically find/replace all the bigrams in the string
    for i in range(0, len(lstTokens)): 
        for j in  range(0,len(rep_patterns)):
            lstTokens[i] = re.sub(rep_patterns[j], replacements[j], lstTokens[i]) # replace with bigram representation 

    lstTokens = [tokens.split(" ") for tokens in lstTokens] # convert back to tokenised lists    

  # Look at the most common words, and remove
  if removeTopFrequentTokens > 0:
    setMostFreqWords = term_fd.most_common(removeTopFrequentTokens)
    print(setMostFreqWords)
    lstMostFreqWordsKeys = sorted(list(map(lambda x: x[0], setMostFreqWords)))
    lstTokens = list(map(lambda x: databasic.filter_by_words_bsearch(x, lstMostFreqWordsKeys), lstTokens))


  # Replace original column with the cleaned data, then write it the names to file
  documentFilePath = featuresDataDir + filePrefix + "_" + colName + "_all.txt"
  df_combined[colName] = list(map(lambda x: ' '.join(x), lstTokens))
  df_combined[[colName]].to_csv(documentFilePath, index=0, header=False)  

  # Remove the original col from the three datasets and then add in the preprocessed version
  del df1[colName]
  del df2[colName]
  del df3[colName]
  df1 = pd.merge(df1, df_combined, on="RowID")
  df2 = pd.merge(df2, df_combined, on="RowID")
  df3 = pd.merge(df3, df_combined, on="RowID")

  # clean up variables from memory
  del df_combined
  del lstSingleWords
  del setSingleWords
  del term_fd
  del words
  del vocab
  del lstTokens

  if removeTopFrequentTokens > 0:    
    del setMostFreqWords

  return df1, df2, df3, documentFilePath


# get a Fast Text language model. Check to see if there is a saved model to use, else train a new one
def getFastTextLangModel(colName, modelFileToUse, modelsDir, filePrefix, documentFilePath, vecDim, forceTrain=False):
  if modelFileToUse == "":
    # check fo a file according to the current file
    modelSavePath = modelsDir + filePrefix + "_" + colName + "_ft_lang_model.model"
  else:
    modelSavePath = modelsDir + modelFileToUse

  modelPath = Path(modelSavePath)
  if modelPath.exists() == False or forceTrain:
    # No model found, or flag passed in to retrain
    fasttext_model = ft.train_unsupervised(documentFilePath, dim=vecDim)
    fasttext_model.save_model(modelSavePath)
  else:
    # Just load the model
    fasttext_model = ft.load_model(modelSavePath)
  
  return fasttext_model


# Given a dataframe and column, and a fast text model, convert that column into a document vector using the 
# inbuilt fasttext function
def convertToDocVectorDataSet(df, colName, fasttext_model):
  doc_vectors = list(map(lambda x: fasttext_model.get_sentence_vector(x), df[colName]))
  df_doc_vectors = pd.DataFrame(doc_vectors)
  df_doc_vectors.columns = list(map(lambda x: colName + "_DocVec_" + str(x), df_doc_vectors.columns))
  df = pd.concat([df[["RowID","BeerID","ReviewerID","rating"]], df_doc_vectors], axis=1)

  del doc_vectors
  del df_doc_vectors

  return df


# Given 3 dataframes (across train, validation and test) and a target column,
# compile the contents of that col across all the frames into one list and convert them all into word token strings that
# can be used to build a language model with fast text. Includes Basic Text Preprocessing
def formatTextColForNLPSupervised(df1, df2, df3, colName, featuresDataDir, filePrefix, removeTopFrequentTokens = 0, setTopBigrams = 0):
  
  # Combine all three datasets into one dataset that is just row ids and the target column
  df_combined = df1[["RowID", colName, "rating"]].append(df2[["RowID", colName, "rating"]]).append(df3[["RowID", colName]])

  # Remove ascii encoding and punctuation
  df_combined[colName] = df_combined.apply(lambda x: str(x[colName]).encode("ascii", "ignore").decode(), axis=1)
  df_combined[colName] = df_combined.apply(lambda x: reutil.str_strip_punctuation(str(x[colName])), axis=1)

  # convert target column into a list of word tokens  
  lstTokens = df_combined.apply(lambda x: str(x[colName]).split(" "), axis=1)
  lstTokens = lstTokens.to_list()

  # Do Text Preprocessing: 
  # remove capitalisation, single letter tokens and stop words
  lstTokens = list(map(lambda x: list(map(lambda y: y.lower(), x)), lstTokens))
  lstTokens = list(map(lambda x: list(filter(lambda y: len(y) >= 2, x)), lstTokens))

  # sort each token list so we can use bsearch
  nltk.download("stopwords")
  stopwordsSorted = sorted(stopwords.words("english"))  
  lstTokens = list(map(lambda x: databasic.filter_by_words_bsearch(x, stopwordsSorted), lstTokens))

  # Create variables for the words and vocab. When we update out tokens lists, we will want to recompile our word list and vocabulary to keep it up to date
  words, vocab = databasic.createWordsAndVocabForTokenLists(lstTokens)

  # Create a term Frequency distribution
  term_fd = nltk.FreqDist(words)

  # remove single occurrence words
  setSingleWords = set(term_fd.hapaxes())
  lstSingleWords = sorted(list(setSingleWords))
  lstTokens = list(map(lambda x: databasic.filter_by_words_bsearch(x, lstSingleWords), lstTokens))
  
  # find the most commong bigrams and join them together to be a single term
  if setTopBigrams > 0:
    bigrams = ngrams(words, n = 2)
    fdbigram = nltk.FreqDist(bigrams)
    mostFreqBigrams = fdbigram.most_common(setTopBigrams)    
    # print(mostFreqBigrams)

    # Convert the bigrams to a list of bigram strings (with spaces)
    rep_patterns = list(map(lambda x: x[0], mostFreqBigrams))
    rep_patterns = list(map(lambda x: x[0] + " " + x[1], rep_patterns))

    # Create another list which is the replacements with the
    replacements = list(map(lambda x: x.replace(" ", "_"), rep_patterns))

    lstTokens = [" ".join(tokens) for tokens in lstTokens] # convert all the token lists back into a single string

    # Loop thought and basically find/replace all the bigrams in the string
    for i in range(0, len(lstTokens)): 
        for j in  range(0,len(rep_patterns)):
            lstTokens[i] = re.sub(rep_patterns[j], replacements[j], lstTokens[i]) # replace with bigram representation 

    lstTokens = [tokens.split(" ") for tokens in lstTokens] # convert back to tokenised lists    

  # Look at the most common words, and remove
  if removeTopFrequentTokens > 0:
    setMostFreqWords = term_fd.most_common(removeTopFrequentTokens)
    print(setMostFreqWords)
    lstMostFreqWordsKeys = sorted(list(map(lambda x: x[0], setMostFreqWords)))
    lstTokens = list(map(lambda x: databasic.filter_by_words_bsearch(x, lstMostFreqWordsKeys), lstTokens))


  # Replace original column with the cleaned data, then write it the names to file
  documentFilePath = featuresDataDir + filePrefix + "_" + colName + "_supervised_all.txt"  

  def getSupLabel(input):
    if input is None or input == "":
      return ""
    else:
      return "__label__" + str(input) + " "

  df_combined[colName] = list(map(lambda x: ' '.join(x), lstTokens))  
  df_combined["Label"] = df_combined.apply(lambda x: getSupLabel(x["rating"]), axis=1)
  df_supervised = df_combined[df_combined["Label"] != ""]
  df_supervised[colName + "Label"] = df_supervised.apply(lambda x: x["Label"] + x[colName], axis=1)
  print(df_supervised[[colName + "Label"]].iloc[0:10, ])

  df_supervised[[colName + "Label"]].to_csv(documentFilePath, index=0, header=False)  

  del df_supervised

  # Remove the original col from the three datasets and then add in the preprocessed version
  del df1[colName]
  del df2[colName]
  del df3[colName]
  df1 = pd.merge(df1, df_combined[["RowID", colName]], on="RowID")
  df2 = pd.merge(df2, df_combined[["RowID", colName]], on="RowID")
  df3 = pd.merge(df3, df_combined[["RowID", colName]], on="RowID")

  # clean up variables from memory
  del df_combined
  del lstSingleWords
  del setSingleWords
  del term_fd
  del words
  del vocab
  del lstTokens

  if removeTopFrequentTokens > 0:    
    del setMostFreqWords

  return df1, df2, df3, documentFilePath  

# get a Fast Text language model. Check to see if there is a saved model to use, else train a new one
def getFastTextLangModelSupervised(colName, modelFileToUse, modelsDir, filePrefix, documentFilePath, vecDim, forceTrain=False):
  if modelFileToUse == "":
    # check fo a file according to the current file
    modelSavePath = modelsDir + filePrefix + "_" + colName + "_ft_lang_model.model"
  else:
    modelSavePath = modelsDir + modelFileToUse

  modelPath = Path(modelSavePath)
  if modelPath.exists() == False or forceTrain:
    # No model found, or flag passed in to retrain
    fasttext_model = ft.train_supervised(documentFilePath, dim=vecDim)
    fasttext_model.save_model(modelSavePath)
  else:
    # Just load the model
    fasttext_model = ft.load_model(modelSavePath)
  
  return fasttext_model  

#########################
### END: FastText NLP Helper Functions
#########################


#########################
### START: Training Model Helper Functions, Surprise and Light
#########################
# Look for an existing saved model to load, otherwise train a surprise model and save it to file
def trainSurpriseModel(algorithm, trainset, modelsDir, filePrefix, modelName, forceTrain=False):
  modelSavePath = modelsDir + filePrefix + "_" + modelName + "_predictor.model"
  modelPath = Path(modelSavePath)
  
  if modelPath.exists() == False or forceTrain:
    # Train the model then Save the predictor model to file
    model = algorithm.fit(trainset)  
    dump.dump(modelSavePath, None, model, True)
  else:
    # load existing model from file
    predictions, model = dump.load(modelsDir + filePrefix + "_" + modelName + "_predictor.model")

  return model


def predictSurpriseModel(modelsDir, filePrefix, modelName, dsName, dataset, dfIds, subrunDir, contentKnnFeats=None):
  # Load the algorithm from the file, the predictions aren't used so that variable will be None
  predictions, algorithm = dump.load(modelsDir + filePrefix + "_" + modelName + "_predictor.model")
  
  # Make Predictions using the model
  NA,valset = train_test_split(dataset, test_size=1.0)

  if contentKnnFeats is not None:
    # For ContentKNN algorithms, we need to pass in the features into the algorithm before using it
    algorithm.setFeatures(contentKnnFeats)

  predictions = algorithm.test(valset)
  
  # Display the MAE
  mae = accuracy.mae(predictions,verbose=False)
  # print("MAE for " + modelName + ": " + str(mae))

  # Convert the Predictions to a dataframe so we can lookup predictions easy
  # uid == BeerId, iid == ReviewerId, r_ui == Original Ration, est = Predicted rating
  lstUIds = list(map(lambda x: x.uid, predictions))
  lstIIds = list(map(lambda x: x.iid, predictions))
  lstTrueRatings = list(map(lambda x: x.r_ui, predictions))
  lstRatingEst = list(map(lambda x: x.est, predictions))
  dfPredictions = pd.DataFrame({ "uid": lstUIds,"iid": lstIIds, "r_ui": lstTrueRatings, "Predict": lstRatingEst })  

  # join the predictions to the ids, sort by rowid and write to out the subrun file
  subRunFilePath = subrunDir + filePrefix + "_" + modelName + "_" + dsName + "_subrun.csv"
  dfPredictions = pd.merge(dfIds, dfPredictions, how="inner", left_on=["BeerID", "ReviewerID"], right_on=["uid", "iid"])
  dfPredictions.sort_values("RowID")[["RowID", "BeerID", "ReviewerID", "Predict"]].to_csv(subRunFilePath, index=False)

  # Clean up the variables from memory
  del predictions
  del algorithm
  del valset
  del lstUIds
  del lstIIds
  del lstTrueRatings
  del lstRatingEst
  del dfPredictions

  return mae

    
def trainLightGbmModel(model, train_feat, train_target, modelsDir, filePrefix, modelName, forceTrain=False):
  modelSavePath = modelsDir + filePrefix + "_" + modelName + "_predictor.model"
  modelPath = Path(modelSavePath)
  
  if modelPath.exists() == False or forceTrain:
    # Train the model and Save the predictor model to file
    model.fit(X=train_feat, y=train_target) 
    model.booster_.save_model(modelsDir + filePrefix + "_" + modelName + "_predictor.model")
  else:
    # load existing model from file
    model = lgb.Booster(model_file=modelsDir + filePrefix + "_" + modelName + "_predictor.model")

  return model


def predictLightGbmModel(vali_ids, vali_feat, vali_target, subrunDir, modelsDir, filePrefix, dsName, modelName):

  mae = 0
  model = lgb.Booster(model_file=modelsDir + filePrefix + "_" + modelName + "_predictor.model")

  predicted = model.predict(vali_feat)
  dfPredicted = pd.DataFrame({"Predict": predicted})

  # join the predictions to the ids, sort by rowid and write to out the subrun file
  subRunFilePath = subrunDir + filePrefix + "_" + modelName + "_" + dsName + "_subrun.csv"
  dfPredicted = pd.concat([vali_ids.reset_index(), dfPredicted], axis=1).drop(columns="index")
  dfPredicted.to_csv(subRunFilePath, index=False)

  if vali_target is not None:      
    mae = mean_absolute_error(vali_target, predicted)
    print("MAE for " + modelName + ": " + str(mae))

  # clean up variables in memory
  del predicted
  del dfPredicted

  return mae

# Version of predict where the inputs are Dask DataFrames rather than Pandas
def predictLightGbmModelDask(vali_ids, vali_feat, vali_target, subrunDir, modelsDir, filePrefix, dsName, modelName):

  mae = 0
  model = lgb.Booster(model_file=modelsDir + filePrefix + "_" + modelName + "_predictor.model")

  predicted = model.predict(vali_feat)

  # join the predictions to the ids, sort by rowid and write to out the subrun file
  subRunFilePath = subrunDir + filePrefix + "_" + modelName + "_" + dsName + "_subrun.csv"
  
  dfPredicted = pd.DataFrame( 
    { "RowID" : vali_ids["RowID"], "BeerID" : vali_ids["BeerID"], 
    "ReviewerID" : vali_ids["ReviewerID"],"Predict": predicted } )
  
  dfPredicted.to_csv(subRunFilePath, index=False)

  if vali_target is not None:      
    mae = mean_absolute_error(vali_target, predicted)
    print("MAE for " + modelName + ": " + str(mae))

  # clean up variables in memory
  del predicted
  del dfPredicted

  return mae  



def trainSkLinearRegModel(model, train_feat, train_target, modelsDir, filePrefix, modelName, forceTrain=False):
  modelSavePath = modelsDir + filePrefix + "_" + modelName + "_predictor.model"
  modelPath = Path(modelSavePath)
  
  if modelPath.exists() == False or forceTrain:
    # Train the model then Save the predictor model to file
    model.fit(train_feat, train_target) 
    pickle.dump(model, open(modelsDir + filePrefix + "_" + modelName + "_predictor.model", 'wb'))   
  else:
    # load existing model from file
    model = pickle.load(open(modelsDir + filePrefix + "_" + modelName + "_predictor.model", 'rb'))

  return model  



def predictSkLinearRegModel(vali_ids, vali_feat, vali_target, 
      subrunDir, modelsDir, filePrefix, dsName, modelName):

  mae = 0
  model = pickle.load(open(modelsDir + filePrefix + "_" + modelName + "_predictor.model", 'rb'))

  predicted = model.predict(vali_feat)
  dfPredicted = pd.DataFrame({"Predict": predicted})

  # join the predictions to the ids, sort by rowid and write to out the subrun file
  subRunFilePath = subrunDir + filePrefix + "_" + modelName + "_" + dsName + "_subrun.csv"
  dfPredicted = pd.concat([vali_ids.reset_index(), dfPredicted], axis=1).drop(columns="index")
  dfPredicted.to_csv(subRunFilePath, index=False)

  # When used on the test data, we have not target/label data. This is just used for evaluation, to 
  # calculate our MAE against real labels. If none is passed in, don't do this
  if vali_target is not None:      
    mae = mean_absolute_error(vali_target, predicted)
    print("MAE for " + modelName + ": " + str(mae))

  # clean up variables in memory
  del predicted
  del dfPredicted  

  return mae
