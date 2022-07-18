#Quelle:https://medium.com/dataseries/how-to-scrape-millions-of-tweets-using-snscrape-195ee3594721
import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import datetime
import os
import json
import statistics as st
from datetime import timedelta, date
import numpy as np 
from textblob import TextBlob
import yfinance as yf
pd.options.mode.chained_assignment = None  
######################################### Gibt den Zeitraum an für den daten erhoben werden sollen. Beachtet feiertage, sowie Wochenenden########
listOfDatesToGetData = []
listOfPublicHolidays = ["2022-04-15","2022-05-30","2022-06-20"]
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
start_dt = date(2022,4,26)
end_dt = date(2022,4,30)
############################## WE raussortieren ################################
weekdays = [6]
for dt in daterange(start_dt, end_dt):
    if dt.weekday() not in weekdays:                    
        listOfDatesToGetData.append(dt.strftime("%Y-%m-%d"))
################## Nach feiertagen prüfen wenn ja dan datum rausschmeißen ################################
# for date in listOfDatesToGetData:
#     if date in listOfPublicHolidays:
#         listOfDatesToGetData.remove(date)
#         print("Holiday: " + date)

################################# Generiert aus array von Daten (Zeitraum) für jeden Tag jeweils einen Datensatz und hängt diesen
################################# der csv im ordner /data FeatureData hinzu
for index, elem in enumerate(listOfDatesToGetData):
    if (index+1 < len(listOfDatesToGetData) and index - 1 >= 0): #Check index bounds
                                    ######     Liste für späteren Dataframe              ########
        tweets_list2 = []
                                    ######                                               #######
                                    ######     Inputdaten für den Scraper und die        #######
                                    ######     yFinace-API um Daten zu generieren.       #######
                                    ######                                               #######
        stock = 'AAPL'
        inter='30m'
        start_date = listOfDatesToGetData[index]
        #start_date="2022-04-25"
        end_date= listOfDatesToGetData[index+1]
        #end_date= "2022-04-26"

        scrape = stock+' since:'+start_date+' until:'+end_date
                                    ######                                               #######
                                    ######     Scrapted Tweets nach AAPL Schlagwort      #######
                                    ######                                               #######
        #NOTIZ: Mit Array arbeiten in dem Strings vom BSP:'AAPL since:2022-05-04 until:2022-05-05' von verschiedenen Tagen drin liegen um werte mit Schleife
        #dann dem enumerate Befehl zu übergeben. Eventuell String aus mehreren Bausteinen zusammenbauen vllt mit Datetim.datetime.date().now() und dann nur Wochentage beachten
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(scrape).get_items()):
            if i == 3500:
                break
            #tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
            tweets_list2.append([tweet.date,tweet.content])
                                    ######                                               #######
                                    ######     Erstellt Dataframe und für 1 Tag zurecht. #######
                                    ######                                               #######
        #NOTIZ: Immer bachten dass das Datum der Loc Bestimmung mit scrape Datum übereinstimmt sonst gibt es einen leren Datensatz  
        # Quelle: https://www.activestate.com/resources/quick-reads/how-to-slice-a-dataframe-in-pandas/                          
        #allTweets_perDay = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
        allTweets_perDay = pd.DataFrame(tweets_list2, columns=['Datetime','Text'])
        endofworkingday = start_date+' 16:00:00+00:00'
        workstart = start_date+' 09:30:00+00:00'
        allTweets_cutAllTweetsOversixtheenOClock = allTweets_perDay.loc[(allTweets_perDay['Datetime']<= endofworkingday)]
        allTweets_perTradingPeriod = allTweets_cutAllTweetsOversixtheenOClock.loc[(allTweets_cutAllTweetsOversixtheenOClock['Datetime']>= workstart)]
        #allTweets_perDay.to_csv("AAPL_Tweets.csv", sep='\t', encoding='utf-8')
                                    ######                                               #######
                                    ######     Entfernt aus einheitlichen Dataframes     #######
                                    ######     alle Links, Hashtags und @Mentions.       #######
                                    ######                                               #######
        for tweet in allTweets_perTradingPeriod['Text']:
                clean_tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
                clean_tweet = re.sub("#","", clean_tweet)
                allTweets_perTradingPeriod.loc[allTweets_perTradingPeriod['Text']== tweet,'Text']=clean_tweet
                                    ######                                               #######
                                    ######     Löscht Rows bei denen kein Tweet mehr     #######
                                    ######     vorhanden ist, da dieser gelöscht ist     #######
                                    ######     weil der Tweet z.B nur aus einem Link     #######
                                    ######     + Mention bestand.                        #######
                                    ######     Fügt Columne "LängeDerTweets" hinzu       #######
                                    ######     aus der später ein Durchschnitt           #######
                                    ######     errechnet wird.                           #######
                                    ######                                               #######                                
        lenghtOfTweetsCharcer = []
        subjectivityScoreOfTweets = []
        politarityScoreOfTweets = []
        positvOrnegativ = []
        for tweet in allTweets_perTradingPeriod['Text']:
                if tweet.isspace():
                    allTweets_perTradingPeriod.drop(allTweets_perTradingPeriod[allTweets_perTradingPeriod['Text']== tweet].index, inplace=True) 
                else:
                    lenghtOfTweetsCharcer.append(len(tweet))
                    blob = TextBlob(tweet)
                    po = blob.sentiment.polarity
                    subjectivityScoreOfTweets.append(blob.sentiment.subjectivity)
                    politarityScoreOfTweets.append(po)
                    if po < 0:
                        positvOrnegativ.append("Negativ")
                    elif po > 0:
                        positvOrnegativ.append("Positiv")
                    elif po == 0:
                        positvOrnegativ.append("Neutral")            
        allTweets_perTradingPeriod = allTweets_perTradingPeriod.assign(charactersPerTweet=lenghtOfTweetsCharcer)
        allTweets_perTradingPeriod = allTweets_perTradingPeriod.assign(subjectivityScoreOfTweet=subjectivityScoreOfTweets)
        allTweets_perTradingPeriod = allTweets_perTradingPeriod.assign(politarityScoreOfTweet=politarityScoreOfTweets)
        allTweets_perTradingPeriod = allTweets_perTradingPeriod.assign(positvOrnegativScore=positvOrnegativ)
                                    ######                                               #######
                                    ######     unterteilt Datafram in 30min kleine       #######
                                    ######     Dataframe-Intevalle                       #######
                                    ######                                               #######
        #NOTIZ: Für die Unterteilung der Intervalle werden von 9:30-16:00 (Öffnungszeiten der amerikanischen Börse) pro Tag 10 random ausgewählte Tweets einem 
        # zeitlichen Intervall von 30min zugewwiese (1 Intervall ist 9:30-10:00). 10 Tweets deshalb weil das eine realistische Menge an Tweets ist die täglich konstant 
        # mit dem Schlagwort AAPL getweetet werden. Es soll verhindert werden dass das LSTM Modell schwächem im Trainig hat,  weil es bei zu großer menge an Tweets in einem Intervall 
        # auch Mal dazu kommen kann dass gar nicht genug Tweets in der Zeit getwittert werden.

        nineToninethirdy = start_date+' 09:30:00+00:00'
        ninethirdyToTen = start_date+' 10:00:00+00:00'
        tenToTenthirdy = start_date+' 10:30:00+00:00'
        tenthirdyToElven = start_date+' 11:00:00+00:00'
        elvenToelventhirdy = start_date+' 11:30:00+00:00'
        elventhirdyToTwelve = start_date+' 12:00:00+00:00'
        twelveToTwelvethirdy = start_date+' 12:30:00+00:00'
        twelvethirdyToThirdy = start_date+' 13:00:00+00:00'
        thirdyToThirdythirdy = start_date+' 13:30:00+00:00'
        thirdythirdyToFourty = start_date+' 14:00:00+00:00'
        fourtyToFourtythirdy = start_date+' 14:30:00+00:00'
        fourtythirdyToFivthen = start_date+' 15:00:00+00:00'
        fivthenToFivethenthirdy = start_date+' 15:30:00+00:00'
        fivthenthirdyToSixteen = start_date+' 16:00:00+00:00'
        timeIntervals = [nineToninethirdy,ninethirdyToTen,tenToTenthirdy,
        tenthirdyToElven,elvenToelventhirdy,elventhirdyToTwelve,twelveToTwelvethirdy,
        twelvethirdyToThirdy,thirdyToThirdythirdy,thirdythirdyToFourty,fourtyToFourtythirdy,
        fourtythirdyToFivthen,fivthenToFivethenthirdy,fivthenthirdyToSixteen]
        dataFramesNames =['firstInterval','secondInterval','thirdInterval','fourthInteval','fiftInterval','sixtInterval','seventhInterval',
        'eigthInterval','nineInterval', 'teenthInterval','eleventhInterval','twelfInterval','thirdtheenInterval']
        d = {}
        counterForCut = 0
        for interval in dataFramesNames:
            cutTweetsInIntervall = allTweets_perTradingPeriod.loc[(allTweets_perTradingPeriod['Datetime']>= timeIntervals[counterForCut])]
            counterForCut = counterForCut + 1
            d[interval] = cutTweetsInIntervall.loc[(cutTweetsInIntervall['Datetime']< timeIntervals[counterForCut])]
        ##################################################################################################################################################################################################################################
        ##################################  VERALTET | VERALTET | VERALTET | VERALTET    #################################################################################################################################################
        ##################################################################################################################################################################################################################################
                                    ######                                               #######
                                    ######     Checkt ob Dataframe-Intervalen mehr       #######
                                    ######     als 10 Datensätze beinhaltet um zu        #######
                                    ######     schauen, ob der Tag geeignet ist          #######
                                    ######     um in den gesamt Datensatz eingehen       #######
                                    ######     zu können und gibt Errormeldung zurück    #######
                                    ######     wenn nicht.(Eventuell noch Boolean für    #######
                                    ######     automatisiertes Skipen des Tages)         #######
                                    ######                                               #######
        # today = datetime.datetime.today()
        # d1 = today.strftime("%b-%d-%Y")
        # for df in d:
        #     if len(df) >= 10:
        #         print('Alle Intervalle haben das Kriterium bestanden und sind somit groß genug einen ')
        #     if len(df) < 10:
        #         print('Interval ',df.name,' vom: ',d1,' hat nicht genug Tweets in seinem Interval um diesen Datensätze dem gesamt Datensatz zuordnen zu können')
        #         #error return, bzw booleanvalue?
        #         break
        #                             ######                                               #######
        #                             ######     Pickt aus bereinigten Datenintervalen     #######
        #                             ######     jeweils 10 Datensätze p. Intervall        #######
        #                             ######     um einheitliche Samplesize zu erstellen.  #######
        #                             ######     Erstellt einen Ordner für einen Tag       #######
        #                             ######     und speichert Datensätze in JSON ab       #######
        #                             ######                                               #######
        # #path = '/Users/Ivan/Desktop/stock-price-predictions-with-lstm-neural-networks-and-twitter-sentiment/data/'+d1 #path for PC
        # path = '/Users/Lenovo/Desktop/stock-price-predictions-with-lstm-neural-networks-and-twitter-sentiment/data/'+d1 #path for Laptop
        # os.mkdir(path)
        # count = 1
        # for df in d:
        #      d[df] = d[df].sample(n=10)
        #      filename = '/Interval '+ str(count)
        #      savepath = path+filename
        #      toJson = d[df].to_json(orient='index')
        #      with open(savepath, "w") as f:
        #         json.dump(toJson, f)
        #      count = count+1
        ####################################################################################################################################################################################################################
        ####################################################################################################################################################################################################################
        ####################################################################################################################################################################################################################
                                    ######                                               #######
                                    ######     Zählt alle Tweets in einem 30min          #######
                                    ######     Intervall und leitet Durchschnitts-       #######
                                    ######     tweets pro 1 Minute ab.                   #######
                                    ######     Errechnet den Max. und Min. Wert für      #######
                                    ######     alle kleinen 1 Minuten "SubIntervalle".   #######
                                    ######     Berechnet den Durchschnittslänge aller    #######
                                    ######     Tweets in einem 30 min Intervall.         #######
                                    ######     Berechnet die Varianz der 1 Min.          #######
                                    ######     Intervalle.                               #######
                                    ######     Erstellt aus diesen Werten einen neuen    #######
                                    ######     Datframe mit den Featurewerten.           #######
                                    ######                                               #######
        listOfDateToSeperateCSV = []
        listInterval = []
        listTotalnumbreOfTweetsPerInterval = []
        listAverageNumbreOfTweetsPerMinutePerInterval = []
        listMinTweetsPerMinute = []
        listMaxTweetsPerMinute = []
        listVarianceOfIntervall = []
        listAverageCharacerPerTweet = []
        listAverageSubjectivityPerTweet =[]
        listAveragePolitarityPerTweet =[]
        listOfNumbreOfPositivTweets = []
        listOfNumbreOfNegativTweets = []
        listShareOfPositiveTweets = []
        listShareOfNegativTweets = []
        for df in d:
            listOfDateToSeperateCSV.append(start_date)
            listInterval.append(df)
            totalnumbreOfTweetsPerInterval = len(d[df]) 
            listTotalnumbreOfTweetsPerInterval.append(totalnumbreOfTweetsPerInterval)
            averageNumbreOfTweetsPerMinutePerInterval = totalnumbreOfTweetsPerInterval/30
            listAverageNumbreOfTweetsPerMinutePerInterval.append(averageNumbreOfTweetsPerMinutePerInterval)
            minutesplit = [g for n, g in d[df].groupby(pd.Grouper(key='Datetime',freq='1Min'))]
            listForCheckingMaxandMin = []
            for split in minutesplit:
                if split.empty:
                    minTweetsPerMinute = 0
                    listForCheckingMaxandMin.append(minTweetsPerMinute)
                else:
                    sizeOfDataFrame = len(split)
                    listForCheckingMaxandMin.append(sizeOfDataFrame)
            listMinTweetsPerMinute.append(min(listForCheckingMaxandMin))
            listMaxTweetsPerMinute.append(max(listForCheckingMaxandMin))
            varianceOfIntervall = np.var(listForCheckingMaxandMin)
            listVarianceOfIntervall.append(varianceOfIntervall)
            averageCharacerPerTweet = st.mean(d[df]['charactersPerTweet'])
            listAverageCharacerPerTweet.append(averageCharacerPerTweet)
            averageSubPerTweet = st.mean(d[df]['subjectivityScoreOfTweet'])
            listAverageSubjectivityPerTweet.append(averageSubPerTweet)
            averagePobPerTweet = st.mean(d[df]['politarityScoreOfTweet'])
            listAveragePolitarityPerTweet.append(averagePobPerTweet)
            positivTweets = []
            positivTweets = d[df][d[df]['positvOrnegativScore'] == 'Positiv']
            numberOfPositivTweets=len(positivTweets)
            listOfNumbreOfPositivTweets.append(numberOfPositivTweets)
            negativTweets = []
            negativTweets = d[df][d[df]['positvOrnegativScore']== 'Negativ']
            numberOfNegativTweets = len(negativTweets)
            listOfNumbreOfNegativTweets.append(numberOfNegativTweets)
            divPoWithTotal = numberOfPositivTweets/totalnumbreOfTweetsPerInterval
            sharePoToTotal = divPoWithTotal * 100
            listShareOfPositiveTweets.append(sharePoToTotal)
            divNegWithTotal = numberOfNegativTweets/totalnumbreOfTweetsPerInterval
            shareNegToTotal = divNegWithTotal * 100
            listShareOfNegativTweets.append(shareNegToTotal)
        featureDatas = pd.DataFrame(list(zip(listOfDateToSeperateCSV,listInterval,listTotalnumbreOfTweetsPerInterval,listAverageNumbreOfTweetsPerMinutePerInterval,listMinTweetsPerMinute,listMaxTweetsPerMinute,listVarianceOfIntervall,
        listAverageCharacerPerTweet, listAverageSubjectivityPerTweet, listAveragePolitarityPerTweet,listOfNumbreOfPositivTweets,listShareOfPositiveTweets,listOfNumbreOfNegativTweets,listShareOfNegativTweets)), 
                                        columns = ['Datum','Interval','Total number of tweets in intervall', 'Average per 1 minute number of tweets in Interval', 'Min. tweets in 1 Minute', 'Max. tweets in 1 Minute', 'Volatility of number of tweets per minute',
                                        'Average character of tweets in Intervall', 'Average subjectivity of tweets in Intervall', 'Average polarity of tweets in Intervall', 'Numbre of all positiv Tweets', 'Share of positiv tweets to total tweets', 'Numbre of all negativ Tweets', 'Share of negativ tweets to total tweets'])
                                    ######                                               #######
                                    ######     Stock Daten über die yfinace-API          #######
                                    ######                                               #######
        stockData = yf.download(stock, start=start_date, end=end_date, interval=inter)
        featureDatas = featureDatas.reset_index(drop=True).merge(stockData.reset_index(drop=True), left_index=True, right_index=True)
        ##path = '/Users/Lenovo/Desktop/stock-price-predictions-with-lstm-neural-networks-and-twitter-sentiment/data/FeatureData.csv' #path for Laptop
        path = '/Users/Ivan/Desktop/stock-price-predictions-with-lstm-neural-networks-and-twitter-sentiment/data/FeatureData.csv' #path for PC
        featureDatas.to_csv(path, mode='a', header=not os.path.exists(path))
        print(featureDatas)
        print("Testdurchlauf fur Tag "+start_date+" erfolgreich abgeschlossen!")    
        ######################   FAZIT: ###############################################################################
        # WAS KANN DAS SCRIPT: Ich kann für einen definierten Tag, über die snscraper Libary mir alle  Tweets nach dem Schlagwort AAPL ziehen, diese in kleine Intervale unterteilen
        # und alle Links, Hashtags und @Mentions entfernen. So definiere Ich mir meinen Datensatz an Tweets.
        # 
        # WAS MUSS DAS SCRIPT NOCH KÖNNEN: 
        # 1. Die Hardgecodeten eingaben der Daten müssen noch durch variablen ewrsetzt werden, sodass man das Script als Object
        # importieren kann und mit einem get Data of Month oder Year Script bedienen kann, um auf einer höheren Ebene über einen Array mehrere Tage nach einander durch das Script
        # laufen lassen kann, um einen großen Datensatz erstellen zu können.
        #
        # 2. Code in Methoden packen um Import aufrufbar machen, Returnwert bei Überprüfung ob 10 vorhanden und Boolean returnen um ganze Tage zu überspringen.
        






