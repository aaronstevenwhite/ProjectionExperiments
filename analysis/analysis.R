####################
## configure session
####################

root.dir <- '~/experiments/ProjectionExperiments/'
analysis.dir <- paste(root.dir, 'analysis/', sep='')
plots.dir <- paste(analysis.dir, 'plots/', sep='')

setwd(analysis.dir)

###############
# load packages
###############

## hierarchical clustering
library(ape)

## multidimensional scaling
library(MASS)

## heteroscedasticity tests
library(lmtest)

## plotting
library(ggplot2)
library(RColorBrewer)
library(tikzDevice)
library(xtable)

## normalization functions (ridit and zscore)
source('normalizing_functions.R')

####################
# configure packages
####################

## set ggplot theme to black and white
theme_set(theme_bw())

###########
# load data
###########

source('filter.R')

library(dplyr)

##############
# frame graphs
##############

## compute verb hierarchical clustering
frame.hiero.verb <- hclust(dist(cast(frame, verb~frame, mean, value='response')))

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'framerawhierotreeverb.tikz', sep=''), width=6, height=5)
# plot(as.phylo(as.hclust(reorder(as.dendrogram(frame.hiero.verb), 30:1, mean))), direction="leftwards")
# dev.off()

## compute frame hierarchical clustering
frame.hiero.frame <- hclust(dist(cast(frame, frame~verb, mean, value='response')))

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'framerawhierotreeframe.tikz', sep=''), width=6, height=5)
# plot(as.phylo(as.hclust(as.dendrogram(frame.hiero.frame))), direction="leftwards", cex=.9)
# dev.off()

## create verb ordering based on hierarchical clustering
verb.order <- frame.hiero.verb$labels[frame.hiero.verb$order]
verb.order <- rev(verb.order[c(5:30, 1:4)])
frame$verb.ordered <- ordered(frame$verb, levels=verb.order)

## create frame ordering based on hierarchical clustering
frame.order <- rev(frame.hiero.frame$labels[frame.hiero.frame$order])
frame.order <- frame.order[c(15:20, 1:14, 21:30)]
frame$frame.ordered <- ordered(frame$frame, levels=frame.order)

## plot frame heatmap
p.frame.raw <- ggplot(ddply(frame, .(verb.ordered,frame.ordered), summarise, response=mean(response)), aes(x=frame.ordered, y=verb.ordered, fill=response)) + geom_tile(color="white") + scale_fill_gradient2(name='Median\nresponse', low="white", high="black") + scale_x_discrete(name=element_blank()) + scale_y_discrete(name=element_blank()) + theme(axis.text.x = element_text(angle=45, hjust=1, vjust=1, size=6), axis.text.y = element_text(size=8), axis.ticks=element_blank(), panel.background=element_blank(), panel.grid=element_blank(), panel.border=element_blank(), legend.background=element_rect(color="black"), legend.position="none")

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'framerawhieroheat.tikz', sep=''), width=6, height=4)
# p.frame.raw
# dev.off()

## compute PCA and SVD
frame <- ddply(frame, .(frame), transform, response.centered=(response-mean(response))/sd(response))
pca.verb <- princomp(as.matrix(cast(frame, verb~frame, mean, value='response.centered')))
svd.verb <- svd(as.matrix(cast(frame, verb~frame, mean, value='response.centered')))

## extract scores (verbs) for components 1 and 2 from PCA
verb.space <- as.data.frame(pca.verb$scores[,c('Comp.1', 'Comp.2')])

## plot verbs on components 1 and 2 from PCA
p.frame.pca.verb <- ggplot(verb.space, aes(x=Comp.1, y=Comp.2, label=rownames(verb.space))) + geom_hline(yintercept=0, alpha=.5) + geom_vline(xintercept=0, alpha=.5) + geom_text(size=3) + scale_x_continuous(name='Principal Component 1', limits=c(-2, 2.5)) + scale_y_continuous(name='Principal Component 2', limits=c(-4.5, 2)) + theme(axis.text.x = element_blank(), axis.title.x = element_text(size=10), axis.text.y = element_blank(), axis.title.y = element_text(size=10), axis.ticks=element_blank())

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'framerawpcaverb.tikz', sep=''), width=6, height=4)
# p.frame.pca.verb
# dev.off()

## extract loadings (frames) for components 1 and 2 from PCA
frame.space <- as.data.frame(pca.verb$loadings[,c('Comp.1', 'Comp.2')])

## adjust loadings slightly to avoid overplotting
frame.space['NP Ved NP about NP','Comp.2'] <- frame.space['NP Ved NP about NP','Comp.2'] - .02
frame.space['It Ved NP WH S','Comp.2'] <- frame.space['It Ved NP WH S','Comp.2'] - .01
frame.space['NP Ved WH to VP','Comp.2'] <- frame.space['NP Ved WH to VP','Comp.2'] + .015
frame.space['S, NP Ved','Comp.2'] <- frame.space['S, NP Ved','Comp.2'] + .01
frame.space['NP Ved there to VP','Comp.2'] <- frame.space['NP Ved there to VP','Comp.2'] + .01

## plot frames on components 1 and 2 from PCA
p.frame.pca.frame <- ggplot(frame.space, aes(x=Comp.1, y=Comp.2, label=rownames(frame.space))) + geom_hline(yintercept=0, alpha=.5) + geom_vline(xintercept=0, alpha=.5) + geom_text(size=2)+ scale_x_continuous(name='Principal Component 1', limits=c(-.4, .35)) + scale_y_continuous(name='Principal Component 2') + theme(axis.text.x = element_blank(), axis.title.x = element_text(size=10), axis.text.y = element_blank(), axis.title.y = element_text(size=10), axis.ticks=element_blank())

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'framerawpcaframe.tikz', sep=''), width=6, height=4)
# p.frame.pca.frame
# dev.off()

## extract scores (verbs) for components 3 and 4 from PCA
verb.space2 <- as.data.frame(pca.verb$scores[,c('Comp.3', 'Comp.4')])

## plot verbs on components 3 and 4 from PCA
p.frame.pca.verb2 <- ggplot(verb.space2, aes(x=Comp.3, y=Comp.4, label=rownames(verb.space2))) + geom_hline(yintercept=0, alpha=.5) + geom_vline(xintercept=0, alpha=.5) + geom_text(size=3) + scale_x_continuous(name='Principal Component 3') + scale_y_continuous(name='Principal Component 4') + theme(axis.text.x = element_blank(), axis.title.x = element_text(size=10), axis.text.y = element_blank(), axis.title.y = element_text(size=10), axis.ticks=element_blank())

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'framerawpcaverb2.tikz', sep=''), width=6, height=4)
# p.frame.pca.verb2
# dev.off()

## extract loadings (frames) for components 3 and 4 from PCA
frame.space2 <- as.data.frame(pca.verb$loadings[,c('Comp.3', 'Comp.4')])

## plot frames on components 3 and 4 from PCA
p.frame.pca.frame2 <- ggplot(frame.space2, aes(x=Comp.3, y=Comp.4, label=rownames(frame.space2))) + geom_hline(yintercept=0, alpha=.5) + geom_vline(xintercept=0, alpha=.5) + geom_text(size=2)+ scale_x_continuous(name='Principal Component 3') + scale_y_continuous(name='Principal Component 4') + theme(axis.text.x = element_blank(), axis.title.x = element_text(size=10), axis.text.y = element_blank(), axis.title.y = element_text(size=10), axis.ticks=element_blank())

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'framerawpcaframe2.tikz', sep=''), width=6, height=4)
# p.frame.pca.frame2
# dev.off()

#########################
## likert factor analysis
#########################

waic <- read.csv(paste(analysis.dir, 'model/likert_factor_analysis/waic_discrete', sep=''), header=FALSE)
names(waic) <- c('Number of features', 'LPPD', 'WAIC')
waic[['LPPD']] <- -2*waic[['LPPD']]
waic.melt <- melt(waic, id='Number of features')
names(waic.melt)[2] <- 'Measure'

p.waic <- ggplot(waic.melt, aes(x=`Number of features`, y=value, linetype=Measure)) + geom_line(size=1.5) + scale_y_continuous(name='', breaks=seq(18000, 24000, 1000)) + scale_x_continuous(breaks=1:15) + scale_linetype_manual(values=c(6,1)) + theme(legend.justification=c(1,0), legend.position=c(1,.8), legend.background=element_rect(color="black"))

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'waic.tikz', sep=''), width=6, height=4)
# p.waic
# dev.off()

## load features and loadings
verb.features <- read.csv(paste(analysis.dir, 'model/likert_factor_analysis/discrete/verbfeatures_14.csv', sep=''), sep=";")
feature.loadings <- read.csv(paste(analysis.dir, 'model/likert_factor_analysis/discrete/featureloadings_14.csv', sep=''), sep=";")

## label features
num.of.features <- nrow(verb.features)  
verb.features$feature <- 1:num.of.features

names(feature.loadings) <- gsub('\\.', ' ', gsub('\\.\\.', ',.', names(feature.loadings)))
feature.loadings$feature <- 1:num.of.features

## melt
verb.features <- melt(verb.features, id='feature')
names(verb.features) <- c('feature', 'verb', 'value')

feature.loadings <- melt(feature.loadings, id='feature')
names(feature.loadings) <- c('feature', 'frame', 'value')

## create feature order

#### feature count-based order
feature.counts <- ddply(verb.features, .(feature), summarise, featuresum=sum(value))
feature.order <- feature.counts$feature[order(feature.counts$featuresum)]

#### loading gini-based order
#feature.gini <- summarize(group_by(feature.loadings, feature), 
#                          gini=ineq(value, type="Gini"))
#feature.order <- feature.gini[order(feature.gini$gini),'feature']

## order verbs and frames
verb.features$verb.ordered <- ordered(verb.features$verb, levels=verb.order)

feature.loadings$frame.ordered <- ordered(feature.loadings$frame, levels=rev(frame.order))
feature.loadings$feature <- as.factor(feature.loadings$feature)

## order features
#verb.features$feature <- as.factor(verb.features$feature)
verb.features$feature.ordered <- ordered(verb.features$feature, levels=feature.order)
feature.loadings$feature.ordered <- ordered(feature.loadings$feature, levels=feature.order)

## plot features
p.verb.features.frame <- ggplot(verb.features, aes(x=feature.ordered, y=verb.ordered, fill=value)) + geom_tile(color="grey") + scale_fill_gradient2(name='Similarity', low="white", high="black") + scale_x_discrete(name=element_blank(), labels=1:num.of.features) + scale_y_discrete(name=element_blank()) + theme(axis.text.x = element_text(size=8), axis.text.y = element_text(size=8), axis.ticks=element_blank(), legend.position="none")

p.feature.loadings.frame <- ggplot(feature.loadings, aes(x=feature.ordered, y=frame, fill=value)) + geom_tile(color="grey") + scale_fill_gradient2(low="white", high="black") + scale_x_discrete(name=element_blank(), labels=1:num.of.features) + scale_y_discrete(name=element_blank()) + theme(axis.text.x = element_text(size=8), axis.text.y = element_text(size=8), axis.ticks=element_blank(), panel.grid=element_blank(), panel.border=element_blank(), legend.position="none") 

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'featureloadingsnonnegative.tikz', sep=''), width=6, height=4)
# p.feature.loadings.frame
# dev.off()

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'verbfeaturesnonnegative.tikz', sep=''), width=6, height=4)
# p.verb.features.frame
# dev.off()

##################
# triad similarity
##################

melt.triad <- function(triad){
  triad$datum <- rownames(triad)
  
  triad.melted <- melt(triad[c('datum', paste('verb', 0:2, sep=''), 'response')], id=c('datum', 'response'))[c('datum', 'response', 'value')]
  triad.melted <- triad.melted[order(triad.melted$datum),]
  triad.melted <- triad.melted[triad.melted$value != triad.melted$response,]
  triad.melted$verbid <- paste('verb', 0:1, sep='')
  
  return(triad.melted)
}

reverse.and.cat <- function(df, reverse){
  df$order <- 'forward'
  
  ind.to.rev <- Vectorize(function(x) which(names(df) == x))(reverse)
  
  if (min(ind.to.rev) > 1){
    indices <- 1:(min(ind.to.rev)-1)
  }
  
  indices <- c(indices, rev(ind.to.rev))
  
  if (max(ind.to.rev) < ncol(df)){
    indices <- c(indices, (max(ind.to.rev)+1):ncol(df))
  }
  
  df.rev <- df[indices]
  names(df.rev) <- names(df)
  
  df.rev$order <- 'backward'
  
  df <- rbind(df, df.rev)
  
  return(df)
}

compute.similarity.triad <- function(triad){
  triad.melted <- melt.triad(triad)
  
  triad.symmetrized <- reverse.and.cat(cast(triad.melted, datum ~ verbid), paste('verb', 0:1, sep=''))
  triad.similarity <- count(triad.symmetrized, paste('verb', 0:1, sep=''))

  return(triad.similarity)
}

similarity.triad <- compute.similarity.triad(triad)

similarity.triad <- rbind(similarity.triad,
                          data.frame(verb0=verb.order, 
                                     verb1=verb.order, 
                                     freq=max(similarity.triad$freq)))

p.triad.raw <- ggplot(similarity.triad, 
                      aes(x=ordered(similarity.triad$verb0, levels=rev(verb.order)), 
                          y=ordered(similarity.triad$verb1, levels=verb.order), 
                          fill=freq)) + geom_tile(color="white") + scale_fill_gradient2(name='Similarity', low="white", high="black", breaks=seq(10,70, 20)) + scale_x_discrete(name=element_blank()) + scale_y_discrete(name=element_blank()) + theme(axis.text.x = element_text(angle=45, hjust=1, vjust=1, size=8), axis.text.y = element_text(size=8), axis.ticks=element_blank(), panel.grid=element_blank(), panel.border=element_blank(), legend.position="none")

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'triadraw.tikz', sep=''), width=6, height=4)
# p.triad.raw
# dev.off()

compute.dist.triad <- function(triad){
  dissimilarity.triad <- compute.similarity.triad(triad)
  dissimilarity.triad$freq <- (length(verb.order)-2)*3 - dissimilarity.triad$freq
  
  dist.triad <- as.dist(as.matrix(cast(dissimilarity.triad, verb0 ~ verb1))[verb.order,verb.order])
  
  return(dist.triad)
}

dist.triad <- compute.dist.triad(triad)
triad.embedding <- as.data.frame(isoMDS(dist.triad)$points)

p.triad.embedding <- ggplot(triad.embedding, aes(x=-V1, y=V2, label=rownames(triad.embedding))) + geom_text(size=3) + theme(axis.text.x = element_blank(), axis.title.x = element_blank(), axis.text.y = element_blank(), axis.title.y = element_blank(), axis.ticks=element_blank())

## uncomment to produce tikz graph
#tikz(paste(plots.dir, 'triadnonmetricmds.tikz', sep=''), width=6, height=4)
#p.triad.embedding
#dev.off()

#####################
## likert similiarity
#####################

compute.similarity.likert <- function(likert){
  likert.symmetrized <- reverse.and.cat(likert, paste('verb', 0:1, sep=''))
  
  similarity.likert <- summarize(group_by(likert.symmetrized, verb0, verb1),
                                 mean.response=mean(response))
  
  return(similarity.likert)
}

similarity.likert <- compute.similarity.likert(likert)
similarity.likert <- rbind(similarity.likert,
                           data.frame(verb0=levels(likert.mean$verb0), 
                                      verb1=levels(likert.mean$verb0), 
                                      mean.response=max(likert.mean$response)))
                           

p.likert.raw <- ggplot(similarity.likert, aes(x=ordered(similarity.likert$verb0, levels=rev(c(verb.order[1:22], 'know', verb.order[23:30]))), y=ordered(similarity.likert$verb1, levels=c(verb.order[1:22], 'know', verb.order[23:30])), fill=mean.response)) + geom_tile(color="white") + scale_fill_gradient2(name='Similarity', low="white", high="black", breaks=seq(1,7)) + scale_x_discrete(name=element_blank()) + scale_y_discrete(name=element_blank()) + theme(axis.text.x = element_text(angle=45, hjust=1, vjust=1, size=8), axis.text.y = element_text(size=8), axis.ticks=element_blank(), panel.grid=element_blank(), panel.border=element_blank(), legend.position="none")

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'likertsimraw.tikz', sep=''), width=6, height=4)
# p.likert.raw
# dev.off()

compute.dist.likert <- function(likert){
  dissimilarity.likert <- compute.similarity.likert(likert)
  dissimilarity.likert$mean.response <- 7 - dissimilarity.likert$mean.response
  
  dist.triad <- as.dist(as.matrix(cast(dissimilarity.likert, verb0 ~ verb1))[c(verb.order, 'know'),c(verb.order, 'know')])
  
  return(dist.triad)
}

dist.likert <- compute.dist.likert(likert)
likert.embedding <- as.data.frame(isoMDS(dist.likert)$points)

p.likert.embedding <- ggplot(likert.embedding, aes(x=V1, y=V2, label=rownames(likert.embedding))) + geom_text(size=3) + theme(axis.text.x = element_blank(), axis.title.x = element_blank(), axis.text.y = element_blank(), axis.title.y = element_blank(), axis.ticks=element_blank())

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'likertnonmetricmds.tikz', sep=''), width=6, height=4)
# p.likert.embedding
# dev.off()

######################
# similarity agreement 
######################

similarity <- merge(filter(similarity.triad[c('verb0', 'verb1', 'freq')], verb0!=verb1), 
                    filter(similarity.likert[c('verb0', 'verb1', 'mean.response')], verb0!=verb1))#, 'ridit', 'zscore')])

m.triad.likert <- rlm(mean.response ~ freq, data=similarity)
bptest(m.triad.likert)

# studentized Breusch-Pagan test
# 
# data:  m.triad.likert
# BP = 135.5431, df = 1, p-value < 2.2e-16

similarity$response.residual <- similarity$mean.response - predict(m.triad.likert)

m.triad.likert.scedastic <- glm(abs(response.residual) ~ freq, family=Gamma(link="inverse"), data=similarity)
similarity$response.residual.standardized <- similarity$response.residual / predict(m.triad.likert.scedastic, type='response')

residual.high <- filter(similarity, response.residual.standardized > 2.5)[c('verb0', 'verb1', 'response.residual.standardized')]
residual.low <- filter(similarity, response.residual.standardized < -2.5)[c('verb0', 'verb1', 'response.residual.standardized')]

residual.high.ordered <- residual.high[rev(order(residual.high$response.residual.standardized)),]
residual.low.ordered <- residual.low[order(residual.low$response.residual.standardized),]

residual.ordered <- rbind(residual.high.ordered, residual.low.ordered)

## uncomment to print tables
# print(xtable(residual.high.ordered[seq(1, nrow(residual.high.ordered), 2),]), include.rownames=FALSE)
# print(xtable(residual.low.ordered[seq(1, nrow(residual.low.ordered), 2),]), include.rownames=FALSE)

#sim.cor <- Vectorize(function(x) cor(similarity$freq, similarity[[x]]))(c('mean.response', 'ridit', 'zscore'))
sim.cor <- cor(similarity$freq, similarity$mean.response)

p.sim.cor <- ggplot(similarity, aes(x=freq, y=mean.response)) + stat_smooth(method="rlm", color="black", size=2, se=F, alpha=.5) + geom_point(alpha=.1) + geom_text(data=merge(similarity, residual.ordered[seq(1, nrow(residual.ordered), 2),c('verb0', 'verb1')]), aes(label=paste(verb0, verb1, sep=' $|$ ')), color="black", size=3) + scale_x_continuous(name='Number of times chosen similar (triad similarity)', limits=c(0,80), breaks=seq(0,80, 10)) + scale_y_continuous('Mean likert similarity', breaks=seq(1,7)) + theme(axis.title.x = element_text(size=10), axis.title.y = element_text(size=10), axis.text.x = element_text(size=8), axis.text.y = element_text(size=8))# + ggtitle('Correlation between triad similarity study and likert scale similarity study')

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'triadlikertraw.tikz', sep=''), width=6, height=4)
# p.sim.cor
# dev.off()

############################
# frame-similarity agreement 
############################

dist.frame <- as.matrix(dist(cast(frame, verb ~ frame, value='response', fun.aggregate = mean)))[verb.order, verb.order]

mantel(dist.frame, as.matrix(dist.triad)[verb.order, verb.order], method='spearman', permutations=10000)
mantel(dist.frame, as.matrix(dist.likert)[verb.order, verb.order], method='spearman', permutations=10000)

dist.frame.triad <- merge(filter(melt(dist.frame), X1!=X2), melt(as.matrix(dist.triad)), by=c('X1', 'X2'))
dist.frame.likert <- merge(filter(melt(dist.frame), X1!=X2), melt(as.matrix(dist.likert)), by=c('X1', 'X2'))

dist.frame.triad$Measure <- 'triad'
dist.frame.likert$Measure <- 'likert'

dist.frame.triad <- transform(dist.frame.triad, frame.norm=value.x/max(value.x), sim.norm=value.y/max(value.y))
dist.frame.likert <- transform(dist.frame.likert, frame.norm=value.x/max(value.x), sim.norm=value.y/max(value.y))

dist.frame.sim <- rbind(dist.frame.triad, dist.frame.likert) 

p.dist.frame.sim <- ggplot(dist.frame.sim, aes(x=frame.norm, y=sim.norm, shape=Measure, linetype=Measure)) + geom_point(alpha=.4, size=1.5) + geom_smooth(color="black", size=1.5, method="loess", family="symmetric") + scale_shape_manual(values = c(1, 2))  + scale_x_continuous(name='Frame distance (normalized)') + scale_y_continuous(name='Dissimilarity (normalized)') + theme(legend.justification=c(1,0), legend.position=c(1,0), legend.background=element_rect(color="black"))

## uncomment to produce tikz graph
# tikz(paste(plots.dir, 'framesimraw.tikz', sep=''), width=6, height=4)
# p.dist.frame.sim
# dev.off()

###############################
## feature-similarity agreement
###############################

## reload verb features 
verb.features <- read.csv(paste(analysis.dir, 'model/likert_factor_analysis/discrete/verbfeatures_14.csv', sep=''), sep=";")