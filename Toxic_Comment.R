setwd("C:/Users/u6335/OneDrive/R/kaggle Toxic comment")

library(data.table)
library(lexicon)
library(tidytext)
library(stringr)
library(rlang)
library(tm)
library(SnowballC)
library(ggthemes)
library(corrplot)
library(caret)
library(ROSE)
library(caTools)
library(textstem)
library(ROCR)
library(xgboost)
library(tictoc)
library(dplyr)

#1 Data Import ==============

train <- fread('train.csv');str(train)
stopwords <- c(sw_buckley_salton, 
               'edit', 'edits', 'wikipedia', 'comment', 'comments', 'wiipedia', 'wij', 'wik', 'wikapidea', 'wiki', 'wiki:', "wiki's", 'wiki2', 'wikia', 'wikiadmins', 'wikiepedia', 'wikiepia') #adding expected common words for Wikipedia edits

swear_words <- readLines('swear_en.txt') #list of swear words from GitHub (+)
bing_clean <- get_sentiments("bing") %>% filter(!word %in% c('envious', 'enviously', 'enviousness')) #removing words with double meaning

#2 Choosing Corpus Words & TF-IDF Analysis============

mystopwords <- data_frame(word = stopwords)

#Words for each type of comment
tf <- 
  train %>% 
  select(id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate) %>% 
  mutate(positive = ifelse(toxic == 0 & severe_toxic == 0 & obscene == 0 & threat == 0 & insult == 0 & identity_hate == 0, 1, 0)) %>% 
  tidyr::gather(key = type, value = measurement, -id, -comment_text) %>%
  filter(measurement != 0) %>%
  unnest_tokens(word, comment_text) %>% 
  mutate(word = stem_words(word))

#in how many comments each word occurs by type
train_tf_comment <- 
  tf %>% 
  count(type, id, word) %>%
  mutate(n = 1) %>% 
  group_by(type) %>% 
  mutate(n_ids = n_distinct(id)) %>% 
  group_by(type, word) %>%
  summarise(tf_comments = sum(n)/max(n_ids), n_comments = sum(n))

#Top-15 words by TF-IDF and who occur in more than 5 comments
train_tf <- 
  tf %>%
  count(type, word) %>% 
  bind_tf_idf(word, type, n) %>% 
  anti_join(stop_words, mystopwords, by = "word") %>% 
  merge(train_tf_comment, by = c('type', 'word')) %>% 
  filter(n_comments >= 3) %>% 
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(type) %>%
  top_n(15, tf_idf) %>% 
  filter(type!='positive' & n_comments >=5)

ggplot(train_tf, aes(word, tf_idf, fill = type)) +
  geom_col(show.legend = FALSE, col = 'black') +
  geom_text(aes(label = n_comments), size = 4, hjust = 1.1, fontface = 'bold')+
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~type, ncol = 2, scale = 'free') +
  coord_flip()+
  labs(title = 'TF-IDF Analysis of comment types', x = '', y = '')+
  theme_bw()
  

#Making a corpus of selected words

words <- sort(unique(c(as.character(train_tf$word), c('nigger', 'nigga', 'painfull'))))
words <- words[nchar(words) > 3 & !words %in% common_names]

tf <- NULL

#3 Data Analysis ===========

model_prep <- function(data, corpus_words){
  tic('total')
  tic('Comment info')
  
  #General information for each comment
  comment_info <- 
    data %>% mutate(nchar = nchar(comment_text), 
                    nletters = str_count(comment_text, '[a-zA-Z]'),
                    ndigits = str_count(comment_text, '[0-9]'),
                    nuppers = ifelse(nletters == 0, 0, str_count(comment_text, '[A-Z]')/nletters), #capslock share
                    nexcl = str_count(comment_text, '[!]')) %>% #quantity of exclamation points
    select(-nletters)
  toc()
  
  tic('Text summary')
  
  #1 Get information about each word in a comment
  #2 Summarise this information for ID by using weighted average
  text_summary <- 
    data %>%
    mutate(comment_text = gsub('[0-9]', '', comment_text),
           comment_text = gsub('[[:punct:]]', '', comment_text)) %>% 
    unnest_tokens(word, comment_text) %>%
    mutate(word = lemmatize_words(word)) %>% 
    count(id, word) %>%
    merge(bing_clean, by = 'word', all.x = T) %>% #marks words as positive or negative -> number of  positive & negative words
    merge(get_sentiments("afinn"), all.x = T) %>% #gives scores [-5:5] to words -> average score
    ungroup() %>% 
    mutate(swear = as.numeric(word %in% swear_words) * n, #if a word is a stopword -> number of stopwords
           stopword = as.numeric(word %in% stopwords) * n, #if a word is a bad/swear word -> number of bad words
           wordlen = nchar(word) * n, #word length -> weighted average word length
           positive = ifelse(!is.na(sentiment) & sentiment == 'positive', 1, 0) * n, 
           negative = ifelse(!is.na(sentiment) & sentiment == 'negative', 1, 0) * n,
           score = score * n) %>% 
    bind_tf_idf(word, id, n) %>% #returns term frequency data
    mutate(tf_idf_cor = tf_idf * n) %>%
    select(-sentiment, -word, -tf, -idf, -tf_idf) %>% 
    dplyr::rename(nwords = n) %>% 
    group_by(id) %>% 
    summarise_all(sum, na.rm = TRUE) %>% #sum variables
    mutate_at(c('swear', 'stopword', 'wordlen', 'positive', 'negative', 'score', 'tf_idf_cor'), funs(./nwords)) %>% #averaging and share calculation
    merge(comment_info, by = 'id', all.y = T) %>% 
    select(-nwords) %>% 
    mutate_at(c('swear', 'stopword', 'wordlen', 'positive', 'negative', 'score', 'tf_idf_cor'), funs(ifelse(is.na(.), 0, .)))
  toc()
  
  tic('Corpus')
  corpus <- data.frame(matrix(0, ncol = length(corpus_words), nrow = nrow(data)))
  colnames(corpus) <- corpus_words
  corpus$comment_text <- data$comment_text
  
  for (n in 1:length(corpus_words)){
    corpus[, n] <- factor(as.numeric(str_detect(tolower(corpus$comment_text), corpus_words[n])))
    #print(paste0('In column ', colnames(corpus)[n], ' was inserted the word ', corpus_words[n], '. Check = ', colnames(corpus)[n] == corpus_words[n]))
  }
  corpus <- mutate(corpus, nigger = factor(ifelse(nigger == 1 | nigga == 1, 1, 0))) %>% select(-licker, -nigga, -painfulli)
  toc()
  toc()
  
  merge(text_summary, corpus, by = 'comment_text') %>% select(-comment_text)
}

modelData <- 
  model_prep(train, words) %>% select(-id) %>% 
  dplyr::rename(pr_toxic = toxic, pr_severe_toxic = severe_toxic, 
                pr_obscene = obscene, pr_threat = threat, 
                pr_insult = insult, pr_identity_hate = identity_hate)

#Positive and Negative comments difference in characters

#Divide comment to positive and negative ones
set.seed(200)
pos_neg <- 
  modelData %>% 
  mutate(meaning = ifelse(pr_toxic == 0 & pr_severe_toxic == 0 & pr_obscene == 0 & 
                            pr_threat == 0 & pr_insult == 0 & pr_identity_hate == 0, 'positive', 'negative')) %>% 
  select(meaning, nchar, ndigits, nuppers, nexcl) %>%
  group_by(meaning) %>% 
  sample_n(3000) #maximum that t-test can take

#Getting average numbers by meaning. Are they really different?
pos_neg %>% group_by(meaning) %>% summarise_all(funs(round(mean(.), 2)))

#Function perform t-test if distibution of both variables is normal and variances are homogenic, and wilcox.test otherwise
#Check for each variable
#Number of vowels in a comment doesn't mean anything
mean_difference <- function(df, val){
  
  t <- df %>% group_by(meaning) %>% summarise_all(funs(round(mean(.), 2))) %>% pull(!!sym(val))
  
  norm_check <- df %>% group_by(meaning) %>% 
    summarise(shtest = shapiro.test(!!sym(val))$p.value > 0.05) #Check for Normal DIstribution (Shapiro Test)
  
  f <- as.formula(paste0(val, " ~ ", 'meaning'))
  
  var_check <- bartlett.test(f, df)$p.value #Bartlett test of homogeneity of variances
  
  if (sum(norm_check$shtest) == 2 & var_check > 0.05){ #t-test if both restrictions are satisfied
    test = t.test(f, df, var.equal = T)
  }
  
  else{
    test = wilcox.test(f, df) #elseway Wilcoxon test
  }
  #m <- paste0(test$method, ' was used. ')
  if (test$p.value < 0.05) { 
    print(paste0(toupper(val), '. Significant difference. Neg = ', t[1], ', Pos = ', t[2], '.'))
  } else {
    print(paste0(toupper(val), '. No significant difference'))
  }
}
for (val in names(pos_neg)[-1]){
  mean_difference(pos_neg, val)
}

#AFINN Scores Analysis
afinn_data <- 
  modelData %>% select(score, pr_toxic, pr_severe_toxic, pr_obscene, pr_threat, pr_insult, pr_identity_hate) %>% 
  mutate(id = row_number()) %>%
  mutate_at(c('pr_toxic', 'pr_severe_toxic', 'pr_obscene', 'pr_threat', 'pr_insult', 'pr_identity_hate'), funs(. * score)) %>% 
  select(-score) %>% 
  tidyr::gather(comment_type, score, -id) %>% 
  filter(score!=0) %>% 
  mutate(comment_type = ordered(comment_type, levels = c('pr_toxic', 'pr_severe_toxic', 'pr_obscene', 
                                                         'pr_threat', 'pr_insult', 'pr_identity_hate'),
                        labels = c('toxic', 'severe', 'obscene', 'threat', 'insult', 'id hate'))) %>% 
  group_by(comment_type) %>% 
  sample_n(400)
  
  
ggplot(afinn_data, aes(x = score, fill = comment_type)) +
  geom_histogram(col = 'black', bins = 50) +
  scale_x_continuous(name = '', limits = c(-3, 1)) +
  labs(y = '', title = 'AFINN Mean Score Histograms by Comment type') +
  scale_fill_discrete(guide = FALSE) +
  facet_grid(comment_type ~ .) +
  theme_minimal() +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank())
  
#Correlation Analysis For Word Meaning Variables

c <- cor(select(modelData, swear, stopword, wordlen, positive, negative, score, tf_idf_cor))

ctest <- cor.mtest(select(modelData, swear, stopword, wordlen, positive, negative, score, tf_idf_cor), 
                   conf.level = .95)

corrplot(c, type = "lower", order = "hclust", tl.col = "black", tl.srt = 360,  insig = "label_sig",
         p.mat = ctest$p, pch.cex = 1.5, pch.col = "black")

#Correlation Analysis For Dependent Variables

c <- cor(train %>% dplyr::select(toxic, severe_toxic, obscene, threat, insult, identity_hate))

corrplot.mixed(c, tl.col = "black", order = "FPC")

#4 Modelling ================

modelling <- function(x_data, y_data, varname, cor_var, to_predict, n = 0, corr = FALSE){
  tic('total')
  data_prep <- function(x_data, y_data, varname, n = 0, cor_var, corr = FALSE){
    if (corr){
      data <- cind(x_data, cor_var)
    } else {
      data <- x_data
    }
    data$pred <-  y_data[, varname]
    levels(data$pred) <- make.names(levels(data$pred))
    
    #Split
    set.seed(123)
    split <- sample.split(data$pred, SplitRatio = 0.7)
    trainData <- subset(data, split==TRUE)
    
    #Proprocess
    preProc <- preProcess(trainData)
    trainData <- predict(preProc, trainData)
    testData <- predict(preProc, subset(data, split==FALSE))
    
    #Balance
    if (n == 0){
      trainData <- ROSE(pred ~ ., data = trainData, seed = 1)$data
    } else {
      set.seed(123)
      trainData <- group_by(trainData, pred) %>% sample_n(n) %>% ungroup
    }
    list(train = trainData, test = testData, pP = preProc)
  }
  
  Log_Reg <- function(trainData, testData, to_predict){
    model <- train(pred ~ ., data = trainData, method="glm", 
                   trControl = trainControl(method="cv", number = 10, classProbs = TRUE))
    Accuracy <- confusionMatrix(predict(model, testData), testData$pred)$overall["Accuracy"]
    AUC <- as.numeric(performance(ROCR::prediction(predict(model, testData, type = 'prob')[, 2], testData$pred), "auc")@y.values)
    prediction <- predict(model, to_predict, type = 'prob')[,2]
    list(Metrics = c(Accuracy, AUC = AUC), Prediction = prediction,
         Model = model)
  }
  
  XGB <- function(trainData, testData, to_predict){
    check <- testData
    trainData <- xgb.DMatrix(data = data.matrix(dplyr::select(trainData, -pred)), label= as.numeric(trainData$pred)-1)
    testData <- xgb.DMatrix(data = data.matrix(dplyr::select(testData, -pred)), label = as.numeric(testData$pred)-1)
    to_predict <- xgb.DMatrix(data = data.matrix(to_predict))
    model <- xgb.train(data = trainData, max.depth = 7, eta = .1, nthread = 2, nround = 20, 
                       watchlist = list(train = trainData, test = testData), objective = "binary:logistic", maximize = FALSE, verbose = 0)
    t <- table(as.numeric(predict(model, testData) > 0.5), as.numeric(check$pred)-1)
    Accuracy <- (t[1] + t[4])/sum(t)
    AUC <- as.numeric(performance(ROCR::prediction(predict(model, testData), as.numeric(check$pred)-1), "auc")@y.values)
    prediction <- predict(model, to_predict)
    list(Metrics = c(Accuracy = Accuracy, AUC = AUC), Prediction = prediction,
         Model = model)
  }
  
  data <- data_prep(x_data, y_data, varname, n = 0, prediction, corr = FALSE)
  print('Data formation is done')
  
  set.seed(123)
  LR <- Log_Reg(data$train, data$test, x_data)
  print('LR is calculated')
  
  set.seed(123)
  XGB <- XGB(data$train, data$test, x_data)
  print('XGB is calculated')
  
  condition <- round(XGB$Metrics['AUC'], 2) >= round(LR$Metrics['AUC'], 2)
  if (condition){
    prediction <- XGB$Prediction
    model <- XGB$Model
    type <- 'XGB'
  } else {
    prediction <- LR$Prediction
    model <- LR$Model
    type <- 'LR'
  }
  toc()
  print(paste0(type, ' model was used. AUC = ', round(ifelse(condition, XGB$Metrics[2], LR$Metrics[2]), 2), 
               '. Other is ', round(ifelse(condition, LR$Metrics[2], XGB$Metrics[2]), 2), '. Accuracy = ', 
               round(ifelse(condition, XGB$Metrics[1], LR$Metrics[1]), 2), 
               '. Other is ', round(ifelse(condition, LR$Metrics[1], XGB$Metrics[1]), 2)))
  list(Prediction = prediction, preProc = data$pP, Model = model, Type = type)
}

x_data <- select(modelData, -contains('pr_'))
y_data <- select(modelData, contains('pr_')) %>% mutate_all(factor)

toxicTrain <- modelling(x_data, y_data, 'pr_toxic', n = 10705, to_predict = x_data)

cor_var <- data.frame(toxic = toxicTrain$Prediction)

obsceneTrain <- modelling(x_data, y_data, 'pr_obscene', cor_var = cor_var, corr = T, to_predict = x_data)

cor_var <- cor_var %>% mutate(obscene = obsceneTrain$Prediction)

insultTrain <- modelling(x_data, y_data, 'pr_insult', cor_var = cor_var, corr = T, to_predict = x_data)

sevtoxTrain <- modelling(x_data, y_data, 'pr_severe_toxic', to_predict = x_data)

idhTrain <- modelling(x_data, y_data, 'pr_identity_hate', to_predict = x_data)

threatTrain <- modelling(x_data, y_data, 'pr_threat', to_predict = x_data)

#5 Submission =================

submission <- function(data, model){
  data <- predict(model$preProc, data)
  if (model$Type == 'LR'){
    predict(model$Model, data, type = 'prob')[, 2]
  } else {
    data <- xgb.DMatrix(data = data.matrix(data))
    predict(model$Model, data)
  }
}

test <- fread('test.csv')

test_data <- model_prep(test, words)

test_x <- test_data[,-1]

toxicTest <- submission(test_x, toxicTrain)
obsceneTest <- submission(mutate(test_x, toxic = toxicTest), obsceneTrain)
insultTest <- submission(mutate(test_x, toxic = toxicTest, obscene = obsceneTest), insultTrain)
sevtoxTest <- submission(test_x, sevtoxTrain)
idhTest <- submission(test_x, idhTrain)
threatTest <- submission(test_x, threatTrain)

submission_data <- data.frame(id = test_data$id, toxic = toxicTest, severe_toxic = sevtoxTest, obscene = obsceneTest,
                              threat = threatTest, insult = insultTest, identity_hate = idhTest)

write.table(submission_data, 'submission.csv', sep = ',', row.names = F, quote = FALSE)


###

#4.1 Toxic Variable Experiment ============

ToxicData <- x_data %>% mutate(toxic = y_data$pr_toxic)
levels(ToxicData$toxic) <- make.names(levels(ToxicData$toxic))

set.seed(123)
split <- sample.split(y_data$pr_toxic, SplitRatio = 0.7)
trainToxic <- subset(ToxicData, split==TRUE)
testToxic <- subset(ToxicData, split==FALSE)

table(trainToxic$toxic)

trainToxicPreproc <- preProcess(trainToxic)
trainToxic <- predict(trainToxicPreproc, trainToxic)
testToxic <- predict(trainToxicPreproc, testToxic)

#Underbalance
trainToxic.under <- ovun.sample(toxic ~ ., data = trainToxic, method = "under", seed = 1)$data
table(trainToxic.under$toxic)

#Different models testing
#Final decision either use LogReg or XGBoost results for the next model by AUC
#The worst results - SVMRadial, DT. SVM Linear is highly correlated with LR and calculates longer. 
#RF is highly correlated with XGBoost and and calculates longer. 
model_effect <- function(model, testdata, y_test){
  acc <- confusionMatrix(predict(model, testdata), y_test)$overall["Accuracy"]
  auc <- as.numeric(performance(prediction(predict(model, testdata, type = 'prob')[, 2], y_test), "auc")@y.values)
  c(acc, AUC = auc)
}

control <- trainControl(method="cv", number = 10, classProbs = TRUE)

set.seed(123)
Toxic.glm <- train(toxic ~ ., data = trainToxic.under, method="glm", trControl=control)

set.seed(123)
Toxic.svml <- train(toxic ~ ., data = trainToxic.under, method="svmLinear", trControl=control)

set.seed(123)
Toxic.svmR <- train(toxic ~ ., data = trainToxic.under, method="svmRadial", trControl=control)

set.seed(123)
Toxic.DT <- train(toxic ~ ., data = trainToxic.under, method="rpart", trControl=control, tuneGrid = expand.grid(.cp = seq(0.01,0.5,0.01)))

set.seed(123)
Toxic.RF <- train(toxic ~ ., data = trainToxic.under, method="rf", trControl=control)

#XGBoost
dtrain <- xgb.DMatrix(data = data.matrix(trainToxic.under[, 1:ncol(trainToxic.under)-1]), label= as.numeric(trainToxic.under$toxic)-1)
dtest <- xgb.DMatrix(data = data.matrix(testToxic[, 1:ncol(trainToxic.under)-1]), label = as.numeric(testToxic$toxic)-1)

set.seed(123)
Toxic.XGB <- xgb.train(data = dtrain, max.depth = 7, eta = .1, nthread = 2, 
                       nround = 50, watchlist = list(train = dtrain, test = dtest), objective = "binary:logistic", maximize = FALSE)

t <- table(pred = as.numeric(predict(Toxic.XGB, dtest) > 0.5), fact = as.numeric(testToxic$toxic)-1)

xgb.plot.importance(xgb.importance(colnames(trainToxic.under), model = Toxic.XGB))

print('Log Regression: '); model_effect(Toxic.glm, testToxic, testToxic$toxic)
print('SVM Linear: '); model_effect(Toxic.svml, testToxic, testToxic$toxic)
print('SVM Radial: '); model_effect(Toxic.svmR, testToxic, testToxic$toxic)
print('Decision Tree: '); model_effect(Toxic.DT, testToxic, testToxic$toxic)
print('Random Forest: '); model_effect(Toxic.RF, testToxic, testToxic$toxic)
print('XGBoost: '); c(Accuracy = (t[1]+t[4])/sum(t), AUC = as.numeric(performance(prediction(predict(Toxic.XGB, dtest), testToxic$toxic), "auc")@y.values))

#Correlation between predictions
PredictionDFToxic <- data.frame(toxic = testToxic$toxic, 
                                LogReg = predict(Toxic.glm, testToxic, type = 'prob')[,2],
                                SVMLinear = predict(Toxic.svml, testToxic, type = 'prob')[,2],
                                SVMRadial = predict(Toxic.svmR, testToxic, type = 'prob')[,2],
                                DecisionTree = predict(Toxic.DT, testToxic, type = 'prob')[,2],
                                RandomForest = predict(Toxic.RF, testToxic, type = 'prob')[,2],
                                XGBoost = predict(Toxic.XGB, dtest))

c <- cor(mutate(PredictionDFToxic, toxic = as.numeric(toxic)-1))

ctest <- cor.mtest(mutate(PredictionDFToxic, toxic = as.numeric(toxic)-1), conf.level = .95)

corrplot(c, type = "lower", order = "hclust", tl.col = "black", tl.srt = 360,  insig = "label_sig",
         p.mat = ctest$p, pch.cex = 1.5, pch.col = "black")



