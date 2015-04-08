###############
# load packages
###############

## correlations
library(Hmisc)
library(irr)
library(vegan)

## data manipulation
library(plyr)
#library(dplyr)
library(reshape)

## set data directory path
root.dir <- dirname(getwd())
data.dir <- paste(root.dir, 'data', sep='/')
materials.dir <- paste(root.dir, 'materials', sep='/')

###################
## define functions
###################

## within group correlation

compute.pair.statistics <- function(df.sub.cast, stat.func){
  subj.combinations <- combn(names(df.sub.cast)[2:ncol(df.sub.cast)], 2)
  
  compute.pair <- function(i){
    subj1 <- subj.combinations[1,i]
    subj2 <- subj.combinations[2,i]
    
    stat.func(df.sub.cast[c(subj1, subj2)])
  }
  
  correl <- sapply(1:ncol(subj.combinations), compute.pair)
  
  return(cbind(as.data.frame(t(subj.combinations)), correl))
}

compute.correlations.in.group <- function(df, g, method=NA){
  df.sub <- subset(df, group==g)
  df.sub.cast <- cast(droplevels(df.sub), item ~ subj, value='response')  
  
  if (is.na(method)){
    stat <- kappam.fleiss(as.matrix(df.sub.cast))$value 
  } else {
    stat <- compute.pair.statistics(df.sub.cast, method) 
  }
  
  return(stat)
}

## subject exclusion

find.outlier.subjects <- function(corrs){
  ## get first quartile and interquartile range
  correl.q1 <- quantile(corrs$correl, .25)
  correl.iqr <- IQR(corrs$correl)
  
  ## find outliers (Tukey's method)
  outlier.comparisons <- subset(corrs, correl < (correl.q1 - 1.5*correl.iqr))
  
  ## find total number of comparisons for subject
  ## and number of comparisons for subject which were outliers
  freq.total <- count(c(as.character(corrs$V1), as.character(corrs$V2)))
  freq.outlier <- count(c(as.character(outlier.comparisons$V1), 
                          as.character(outlier.comparisons$V2)))

  ## rename columns
  names(freq.total) <- c('subj', 'total')
  names(freq.outlier) <- c('subj', 'outlier')
  
  ## find outliers 
  outlier.subjects <- subset(merge(freq.total, freq.outlier), 
                             outlier == total)$subj
  
  return(outlier.subjects)
}

##################
## load frame data
##################

## set frame data path
frame.path <- paste(data.dir, 'frame/frame.preprocessed', sep='/')
frame.key.path <- paste(materials.dir, 'frame/frame.key', sep='/')

## read frame data
frame <- read.csv(frame.path)

## load frame key (does not include degree frames)
frame.key <- read.csv(frame.key.path)

## merge frame data and key (removes degree data)
frame <- merge(frame.key, frame, by.x='frame.old', by.y='frame')

######################
## validate frame data
######################

## add group number for participants
frame <- ddply(frame, .(subj), transform, group=min(item))

## find unique groups
groups <- unique(frame$group)

## compute cohens kappa for all pairs in all groups
spearmans.rho <- function(x) cor(x[,1], x[,2], method="spearman")
frame.spearmans <- do.call('rbind', 
                           lapply(groups, 
                                  function(g) compute.correlations.in.group(frame,
                                                                            g, 
                                                                            method=spearmans.rho)))
## exclude outlier subjects
frame.outlier.subjects <- find.outlier.subjects(frame.spearmans)
frame <- droplevels(subset(frame, !(subj %in% frame.outlier.subjects)))

## uncomment to write frame data
# write.table(frame[c('subj', 'item', 'verb', 'frame', 'response')], 
#             paste(data.dir, 'frame/frame.filtered', sep='/'),
#             sep=';',
#             quote=F,
#             row.names=F)

##################
## load triad data
##################

## set triad data path
triad.path <- paste(data.dir, 'triad/triad.preprocessed', sep='/')

## read triad data
triad <- read.csv(triad.path)

######################
## validate triad data
######################

## add group number for participants
triad <- ddply(triad, .(subj), transform, group=min(item))

## find unique groups
groups <- unique(triad$group)

## compute fleiss kappa for all groups
triad.fleiss <- sapply(groups, function(g) compute.correlations.in.group(triad, g))

## compute cohens kappa for all pairs in all groups
cohens.kappa <- function(x) kappa2(x, 'unweighted')$value
triad.cohens <- do.call('rbind', 
                        lapply(groups, 
                               function(g) compute.correlations.in.group(triad, 
                                                                         g, 
                                                                         method=cohens.kappa)))

## exclude outlier subjects
triad.outlier.subjects <- find.outlier.subjects(triad.cohens)
triad <- droplevels(subset(triad, !(subj %in% triad.outlier.subjects)))

## uncomment to write triad data
# write.csv(triad[c('subj', 'verb0', 'verb1', 'verb2', 'responseindex')], 
#           paste(data.dir, 'triad/triad.filtered', sep='/'),
#           quote=F,
#           row.names=F)

###################
## load likert data
###################

## set likert data path
likert.path <- paste(data.dir, 'likert/likert.preprocessed', sep='/')

## read likert data
likert <- read.csv(likert.path)

#######################
## validate likert data
#######################

## add group number for participants
likert <- ddply(likert, .(subj), transform, group=min(item))

## find unique groups
groups <- unique(likert$group)

## compute cohens kappa for all pairs in all groups
likert.spearmans <- do.call('rbind', 
                           lapply(groups, 
                                  function(g) compute.correlations.in.group(likert,
                                                                            g, 
                                                                            method=spearmans.rho)))
## exclude outlier subjects
likert.outlier.subjects <- find.outlier.subjects(likert.spearmans)
likert <- droplevels(subset(likert, !(subj %in% likert.outlier.subjects)))

## uncomment to write frame data
# write.csv(likert[c('subj', 'verb0', 'verb1', 'response')], 
#           paste(data.dir, 'likert/likert.filtered', sep='/'),
#           quote=F,
#           row.names=F)