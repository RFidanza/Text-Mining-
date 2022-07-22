#carico le recensioni
firestick <- read.csv('Amazon_Review___Complete_Data__669552874.csv',sep = ',')

#installo le librerie

install.packages('udpipe')
install.packages('tm')
install.packages('wordcloud')
install.packages('RColorBrewer')
install.packages('ggplot2')
install.packages('sentimentr')
install.packages('data.table')
install.packages('BTM')
install.packages('textplot')
install.packages('concaveman')
install.packages('ggraph')
install.packages("textrank")

library(textrank)
library(ggraph)
library(BTM)
library(udpipe)
library(sentimentr)
library(data.table)
library(tm)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)
library(textplot)
library(concaveman) 

#PULIZIA DEL TESTO
firestickfrasi<- get_sentences(firestick$Review.Content)
firestickcorpus<- VCorpus(VectorSource(firestickfrasi))

#trasformo tutto in minuscolo
firestick_pulito<-tm_map(firestickcorpus,content_transformer(tolower))
#rimozione delle stopwords
firestick_pulito<-tm_map(firestick_pulito,removeWords,stopwords('english'))
#rimozione della punteggiatura
firestick_pulito<-tm_map(firestick_pulito,removePunctuation)
#rimozione spazi bianchi
firestick_pulito<-tm_map(firestick_pulito,stripWhitespace)
#rimozione dei numeri
firestick_pulito<-tm_map(firestick_pulito,removeNumbers)
#stemming
firestick_pulito<-tm_map(firestick_pulito,stemDocument)

#trasformo in matrice
firestickpulitomatrice<- TermDocumentMatrix(firestick_pulito)
matrice<- as.matrix(firestickpulitomatrice)

#ordinamento decrescente
matricefr<-sort(rowSums(matrice),decreasing = TRUE)
matriced<-data.frame(word=names(matricefr),freq=matricefr)

#stampo le prime 10 parole più frequenti
head(matriced,10)

#grafico per le 10 parole più frequenti
barplot(matriced[1:10,]$freq,names.arg = matriced[1:10,]$word,
        col = '#F7E89F',main='Top 10 parole più frequenti',
        ylab = 'Frequenza parole')

#worldcloud delle parole più utilizzate
set.seed(1234)
wordcloud(words = matriced$word,freq = matriced$freq,min.freq = 5,
          max.words = 300,random.order = FALSE,rot.per = 0.40,
          colors = brewer.pal(8,'Dark2'))

#pulizia del testo delle stelle
firestick$Rating <- gsub(' out of 5 stars','',firestick$Rating)
#calcolo delle stelle con relativo grafico
stelle<- table(firestick$Rating)
data <- data.frame(
  stars=c("1","2","3","4","5") ,  
  value=c(stelle)
)
ggplot(data, aes(x=stars, y=value)) + 
  geom_bar(stat = "identity")


#calcolo del sentiment per frase
sentences<- sentimentr::get_sentences(firestick$Review.Content)
calcolo_sentiment<- sentiment(sentences)

#grafico del sentiment per ogni frase
ggplot(data = calcolo_sentiment, aes(x=calcolo_sentiment$sentiment))+geom_density(fill='#69b3a2',color='#e9ecef',alpha=0.8)+
  labs(title = 'SENTIMENT PER OGNI FRASE',subtitle = 'Distribuzione empirica',x='Sentiment',y='Densità')+
  theme(plot.title = element_text(hjust = 0.5),plot.subtitle = element_text(hjust = 0.5))

# calcolo del sentiment per ogni recensione
groups<- sentiment_by(sentences)

#grafico del sentiment per ogni recensione
ggplot(data=groups, aes(x=groups$ave_sentiment))+geom_density(fill='#69b3a2',color='#e9ecef',alpha=0.8)+
  labs(title = 'SENTIMENT PER OGNI RECENSIONE',subtitle = 'Distribuzione empirica',x='Sentiment',y='Densità')+
  theme(plot.title = element_text(hjust = 0.5),plot.subtitle = element_text(hjust = 0.5))

#POS-tagging

#scarico il modello
modello <- udpipe_download_model(language = "english")
str(modello)

#carico il modello in una variabile
udmodello <- udpipe_load_model(file = modello$file_model)

#inserisco il testo
testo <- firestick$Review.Content

#annotazione del testo
x <- udpipe_annotate(udmodello,x=testo)
x <- as.data.frame(x)
str(x)

#BTM

#conversione in data table
biterms <- as.data.table(x)

#calcolo cooccorrenze
biterms <- biterms[,cooccurrence(x=lemma,relevant=upos %in% c('NOUN','ADJ','VERB') & nchar(lemma)>2 & !lemma %in% stopwords('en'),skipgram= 3),by = list(doc_id)]

set.seed(123456)
#dati per addestrare il modello
traindata<-subset(x,upos %in% c('NOUN','ADJ','VERB') & !lemma %in% stopwords('en') & nchar (lemma)>2)
traindata <- traindata[,c('doc_id','lemma')]
model <- BTM(traindata, biterms = biterms, k=6 ,iter = 2000, background = TRUE,trace = 100)

#grafico
plot(model, top_n = 10, title= 'BTM model', labels = c('Ordine/Acquisto','Connettività','Servizi','Dispositivo','Feedback installazione','Telecomando'))

#text-rank- summarization

cat(unique(x$sentence),sep="\n")
names(x)
head(x[,c("sentence_id","lemma","upos")],10)
keyw <- textrank_keywords(x$lemma,relevant=x$upos%in%c("NOUN","VERB","ADJ"))
subset(keyw$keywords,ngram>1)

x$textrank_id <- unique_identifier(x,c("doc_id","paragraph_id","sentence_id"))
sentences <- unique(x[,c("textrank_id","sentence")])
head(sentences,10)
terminology <- subset(x,upos%in%c("NOUN","ADJ"))
terminology <- terminology[,c("textrank_id","lemma")]
terminology
#procedura per rank sentences, molto lunga a causa dell'elevato numero di sentences
tr <- textrank_sentences(data=sentences,terminology=terminology)

tr$sentences

tr$sentences_dist

tr$pagerank

vv <- sort(tr$pagerank$vector,decreasing = TRUE)

#grafico rank size, mostra peso delle varie sentences
plot(vv,type = "b",ylab = "Pagerank",main="Textrank")

s <- summary(tr,n=10)

s <- summary(tr,n=5,keep.sentence.order = TRUE)

cat(s,sep="\n")




