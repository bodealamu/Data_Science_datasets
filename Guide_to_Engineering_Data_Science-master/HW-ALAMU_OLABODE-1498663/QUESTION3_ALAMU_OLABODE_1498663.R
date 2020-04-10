#Author: Moadh Mallek
#-----
 

library('pixmap')
setwd("~/Desktop/Guide to data science")
rm(list=ls())
image <- read.pnm('images(1).ppm')

red.matrix <- matrix(image@red, nrow = image@size[1], ncol = image@size[2])
green.matrix <- matrix(image@green, nrow = image@size[1], ncol = image@size[2])
blue.matrix <- matrix(image@blue, nrow = image@size[1], ncol = image@size[2])

# Part 1
image(green.matrix, col = heat.colors(255))

green.matrix.svd <- svd(green.matrix)


# Part to complete question 2)
d <- green.matrix.svd$d
u <- green.matrix.svd$u
v <- green.matrix.svd$v

  
# Recontruct green.matrix question 3)
green.matrix.reconstruction <-u %*% diag(d) %*% t(v)

#SVD
i <- 3
green.matrix.compressed <- u[,1:i] %*% diag(d[1:i]) %*% t(v[,1:i])
image(green.matrix.compressed, col = heat.colors(255))

i <- 5
green.matrix.compressed <- u[,1:i] %*% diag(d[1:i]) %*% t(v[,1:i])
image(green.matrix.compressed, col = heat.colors(255))

i <- 10
green.matrix.compressed <- u[,1:i] %*% diag(d[1:i]) %*% t(v[,1:i])
image(green.matrix.compressed, col = heat.colors(255))

for (i in c(3, 5, 10))
{
  green.matrix.compressed <- u[,1:i] %*% diag(d[1:i]) %*% t(v[,1:i])
  jpeg(paste('xid-28852613_1', i, '.jpg', sep = ''))
  image(green.matrix.compressed, col = heat.colors(255))
  dev.off()
}


## Singular Value of 10 makes the best image
