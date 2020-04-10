#PROBLEM 2
#B
library(Rlof)
attach(outlier_data_edit)
head(outlier_data_edit)
coordinates=lof(outlier_data_edit,10)
print(coordinates)

#C
outliers = order(coordinates,decreasing=T)[1:6]
print(outliers)
N <- nrow(outlier_data_edit)
labels <- 1:N
labels[-outliers] <- "."
biplot(prcomp(outlier_data_edit), cex=.8, xlabs=labels)

pch <- rep(".", N)

pch[outliers] <- "+"

col <- rep("blue", N)


col[outliers] <- "purple"

pairs(outlier_data_edit, pch=pch, col=col)

#D 
k = c(10,20,40,75)
LOF_scores.k = lof(outlier_data_edit,k)
print(LOF_scores.k)
outliers.k = order(LOF_scores.k,decreasing=T)[1:6]
print(outliers.k)

# Plotting
plot.ts(LOF_scores.k, xlab='LOF Scores',ylab='K values',
        main='LOF Scores for Different K values')