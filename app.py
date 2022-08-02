import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import csv

import random
from math import sqrt
from sklearn.decomposition import PCA
import pandas as pd
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from datetime import datetime
from pytrends.request import TrendReq
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
#reg=linear_model.LinearRegression()
from math import sqrt
from flask import Flask,render_template,request,redirect,url_for,session
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])

def index():
    if request.method == 'POST':
        session['brand']=request.form['category']
        os=request.form['os']
        warranty=request.form['warranty']
        memory=request.form['memory']
        processor=request.form['processor']
        res=request.form['Resolution']
        memory_tech=request.form['MemoryTechnology']
        Processor_speed=request.form['proSpeed']
        price=request.form['Price']
        new_feature=request.form['newFeatureInp']
        session["input"] = list()
        dell=apple=averatec=asus=acer=fujitsu=gateway=hp=panasonic=lenovo=toshiba=sony=0
        macos=vistabusiness=vistahb=vistahp=vistault=winxp=winxppro=0
        intelcoreduo=intelcoresingle=celeron=pentium=sempron=turion=athlon=powerpc=0
        ddr2=sdram=ddrram=0
        market_impact=0
        if 'Infrared' in request.form:
            infrared=1
            session["input"].append("Infrared")
            market_impact+=getval(session['brand']+' Infrared')
        else:
            infrared=0
        if 'Bluetooth' in request.form:
            bluetooth=1
            session["input"].append("Bluetooth")
            market_impact+=getval(session['brand']+' Bluetooth')
        else:
            bluetooth=0
        if 'Port Replicator' in request.form:
            port_replicator=1
            session["input"].append("Port Replicator")
            market_impact+=getval(session['brand']+' Port Replicator')
        else:
            port_replicator=0
        if 'Docking Station' in request.form:
            docking_station=1
            session["input"].append("Docking Station")
            market_impact+=getval(session['brand']+' Docking Station')
        else:
            docking_station=0
        if 'Fingerprint' in request.form:
            fingerprint=1
            session["input"].append("Fingerprint")
            market_impact+=getval(session['brand']+' Fingerprint')
        else:
            fingerprint=0
        if 'External Battery' in request.form:
            external_battery=1
            session["input"].append("External battery")
            market_impact+=getval(session['brand']+' External Battery')
        else:
           external_battery=0
        if 'CDMA' in request.form:
            cdma=1
            session["input"].append("CDMA")
            market_impact+=getval(session['brand']+' CDMA')
        else:
            cdma=0
        if 'Subwoofer' in request.form:
            subwoofer=1
            session["input"].append("Subwoofer")
            market_impact+=getval(session['brand']+' Subwoofer')
        else:
            subwoofer=0
            
        #Brand
            
        if session['brand']=='Dell':
            dell=1
            
        if session['brand']=='Apple':
            apple=1
            
        if session['brand']=='Averatec':
            averatec=1
            
        if session['brand']=='Asus':
            asus=1
        if session['brand']=='Acer':
            acer=1

        if session['brand']=='Fujitsu':
            fujitsu=1
        if session['brand']=='Gateway':
            gateway=1
        if session['brand']=='HP':
            hp=1
        if session['brand']=='Panasonic':
            panasonic=1
        if session['brand']=='Toshiba':
            toshiba=1
        if session['brand']=='Sony':
            sony=1
        if session['brand']=='Lenovo':
            lenovo=1                             

        session["input"].append(session["brand"])
      #Operating System

        if os=='Mac OS':
            macos=1
        if os=='Vista Business':
            vistabusiness=1
        if os=='VistaHB: Vista Home Basic':
            vistahb=1
        if os=='VistaHP: Vista Home Premium':
            vistahp=1
        if os=='VistaUlt: Vista Ultimate':
            vistault=1
        if os=='WinXP':
            winxp=1
        if os=='WinXP_Pro':
            winxppro=1
        session["input"].append(os)   
       #Processors

        if processor=='Intel Core2 Duo':
            intelcoreduo=1
        if processor=='Intel Core2 Single':
            intelcoresingle=1
        if processor=='Intel Celeron':
            celeron=1
        if processor=='Intel Pentium Dual-Core':
            pentium=1
        if processor=='AMD Mobile Sempron':
            sempron=1
        if processor=='AMD Turion 64x2':
            turion=1
        if processor=='AMD Athlon 64x2 Doble Nucleo':
            athlon=1
        if processor=='Power PC':
            powerpc=1
        session["input"].append(processor)
        #Memory Technology
            
        if memory_tech=='DDR2':
            ddr2=1
        if memory_tech=='SDRAM':
            sdram=1
        if memory_tech=='DDR-SDRAM(DDRRAM)':
            ddrram=1
        session["input"].append(memory_tech) 
 
        vals={
                  "Max Horizontal Resolution": res,
                  "Installed Memory": memory,
                  "Processor Speed": Processor_speed,
                  "Infrared": infrared,
                  "Bluetooth": bluetooth,
                  "Port Replicator": port_replicator,
                  "Docking Station": docking_station,
                  "Fingerprint": fingerprint,
                  "Subwoofer": subwoofer,
                  "External Battery": external_battery,
                  "CDMA": cdma,
                  "Warranty-Days": warranty,
                  "Memory Technology_DDR-SDRAM_(DDRRAM)":ddrram,
                  "Memory Technology_DDR2":ddr2,
                  "Memory Technology_SDRAM":sdram,
                  "Manufacturer_Acer":acer,
                  "Manufacturer_Apple":apple,
                  "Manufacturer_Asus":asus,
                  "Manufacturer_Averatec":averatec,
                  "Manufacturer_Dell":dell,
                  "Manufacturer_Fujitsu":fujitsu,
                  "Manufacturer_Gateway":gateway,
                  "Manufacturer_HP":hp,
                  "Manufacturer_Lenovo":lenovo,
                  "Manufacturer_Panasonic":panasonic,
                  "Manufacturer_Sony":sony,
                  "Manufacturer_Toshiba":toshiba,
                  "Processor_AMD Athlon 64x2 Doble Nucleo":athlon,
                  "Processor_AMD Mobile Sempron":sempron,
                  "Processor_AMD Turion 64x2":turion,
                  "Processor_Intel Celeron":celeron,
                  "Processor_Intel Core2 Duo":intelcoreduo,
                  "Processor_Intel Core2 Solo":intelcoresingle,
                  "Processor_Intel Pentium Dual-Core":pentium,
                  "Processor_PowerPC":powerpc,
                  "Operating System_Mac_OS":macos,
                  "Operating System_VistaHB:_Vista_Home_Basic":vistahb,
                  "Operating System_VistaHP:_Vista_Home_Premium":vistahp,
                  "Operating System_VistaUlt:_Vista_Ultimate":vistault,
                  "Operating System_Vista_Business":vistabusiness,
                  "Operating System_WinXP":winxp,
                  "Operating System_WinXP_Pro":winxppro,
                  "Price":price,
                  "market_impact": round(market_impact+smarket_impact(session['brand'],new_feature +' Laptop'),2)
                }
        df2=pd.DataFrame([vals])
        df2.to_csv('input.csv', index=False)
        session['new feature'] = new_feature
        session["input"].append(session['new feature']) 
        return redirect(url_for('forecast'))
        #return render_template('index.html')
    else:
        return render_template('index.html')
def getval(a):
    df = pd.read_csv("newdict.csv")
    for index,row in df.iterrows():
     return (row[a])

class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = 'uF83mjqjEB4wKcuM1Oq7BQKfR'
        consumer_secret = 'sZ5emtKEZ3J6jNIhSEUbF40VFBaZPt1O6Jt1dnHk0nMMtgBhfY'
        access_token = '594616639-Jhl4SB2gfizzQ5kIUpaUlwyG2MXb3lEJQFPMlqKe'
        access_token_secret = 'MSL65481uvjY7Csd8Cak8NvvZOtmooiBuMt1is2HQdhdA'
 
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
 
    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
 
    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
 
    def get_tweets(self, query, count = 10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
 
        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q = query, count = count)
 
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
 
                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
 
                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
 
            # return parsed tweets
            return tweets
 
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
 
def tweet(a):
    # creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    tweets = api.get_tweets(query = a, count = 2000)
    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # percentage of positive tweets
    #print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # percentage of negative tweets
    #print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
    if(len(tweets)==0):
        return [1,1]
    else:
        return [float(format(100*len(ptweets)/len(tweets))),float(format(100*len(ntweets)/len(tweets)))]
######
def no_hits(a,y1,m1,d1,y2,m2,d2):
    kw_list = [a]
    start= datetime(y1, m1, d1).date()
    ## FIRST RUN ##
    # Login to Google. Only need to run this once, the rest of requests will use the same session.
    pytrends = TrendReq()
    # Run the first time (if we want to start from today, otherwise we need to ask for an end_date as well
    today = datetime(y2, m2, d2).date()
    # Create new timeframe for which we download data
    frame = start.strftime('%Y-%m-%d')+' '+today.strftime('%Y-%m-%d')
    pytrends.build_payload(kw_list, cat=0, timeframe=frame, geo='', gprop='')
    time_df = pytrends.interest_over_time()
    if(time_df.empty):
        return 0
    else:
        p=time_df[a].tolist()
        s=sum(p) / float(len(p))
        return s
def smarket_impact(a,b):
    t=tweet(a+' '+b)
    h=no_hits(a+' '+b,2015,1,1,2018,8,18)
    if(t[0]==0.0):
        return 0.0
    else:
        return (t[0]/100)*h    
@app.route('/forecast',methods=['GET','POST'])
def forecast():
    

    def cluster():
        data = pd.read_csv("C01.csv")
        clus_data = data.iloc[: , :-1]
        #print(clus_data)
        
        input_data = pd.read_csv("input.csv")
        model = KMeans(n_clusters = 4)
        model.fit(clus_data)
        clussAssignment = model.predict(clus_data)
        
        #these list will have index of each element of respective cluster
        c0 = list()
        c1 = list()
        c2 = list()
        c3 = list()
        for i in range(len(clussAssignment)):
            if(clussAssignment[i] == 0 ):
                c0.append(i)
            elif(clussAssignment[i] == 1 ):
                c1.append(i)
            elif(clussAssignment[i] == 2 ):
                c2.append(i)
            elif(clussAssignment[i] == 3 ):
                c3.append(i)

        #print(len(c0) , len(c1) , len(c2) , len(c3))    
        
        result =model.predict(input_data)
        #print(result)
        pc = list() #this will have predicted cluster in it
        if(result == 0):
            pc = c0
        elif(result == 1):
            pc = c1
        elif(result == 2):
            pc = c2
        elif(result == 3):
            pc = c3
        elif(result == 4):
            pc = c4
        
        data.iloc[pc].to_csv('final_cluster.csv',index=False)

    def train_and_predict():

        clus = pd.read_csv("final_cluster.csv") #cluster predicted by clustring method
        x = clus.iloc[: , :-1].values
        y = clus.iloc[: , -1].values#sales
        x_train , x_test ,y_train , y_test = train_test_split(x , y , test_size=0.3 , random_state=0)
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        rms_value = sqrt(mean_squared_error(y_test, y_pred))
        
        input_x = pd.read_csv("input.csv")
        all_sales = list()
        total_price = input_x['Price'];
        input_x['Price'] = total_price - (total_price*0.2)
        all_sales.append(regressor.predict(input_x))
        
        input_x['Price'] = total_price
        all_sales.append(regressor.predict(input_x))
        input_x['Price'] = total_price + (total_price*0.1)
        all_sales.append(regressor.predict(input_x))
        return all_sales,rms_value
        
        
    def diffusion_model(all_sales):
        p=0.03
        q=0.38
        combined_sum=p+q
        rate_of_immitation=q/p
        three_sales = list()
        for m in all_sales :
            total=0
            sales = list()
            for i in range(12):
                one_month_sale=m*(((combined_sum)**2)/p)*(2.71828**(-combined_sum*i))/((1+(rate_of_immitation*2.71828**(-combined_sum*i)))**2)
                sales.append(one_month_sale)
                total+=one_month_sale
            three_sales.append(sales)
            print(total)
        plot_graph(three_sales)

    def plot_graph(sales):
        import matplotlib.pylab as plt
        plt.plot(range(12) , sales[0], 'og-')
        #plt.annotate("Price reduced by 20%",xy=(6,10000),xytext=(3,1.5))
        #plt.figtext("Price reduced by 20%")
        plt.plot(range(12) , sales[1], 'ob-')
        plt.plot(range(12) , sales[2], 'or-')
        plt.xlabel("Time In Months")
        plt.ylabel("Sales Per Unit")
        plt.savefig("static/graph.png")

    def value1():
        print()

    value1()

    cluster()
    ans,err=train_and_predict()
    diffusion_model(ans)
    max=ans[1]+err
    min=ans[1]-err
    esales = ans[1]
    isales = ans[0]
    rsales = ans[2]
    incr = abs(esales[0]-isales[0])*100/esales[0]
    redu = abs(esales[0]-rsales[0])*100/esales[0]
    session['Maximum']=(int)(round(max[0]))
    session['Minimum']=(int)(round(min[0]))
    results={
           "Min":(int)(round(min[0])),
           "Max":(int)(round(max[0]))
        }
    df2=pd.DataFrame([results])
    print(session["input"])
    df2.to_csv('Result.csv', index=False)
    if min<0:
        return render_template('next.html',data=session["input"],max=(int)(round(max[0])),min=500,imin=incr,rmin=redu)
    else:
        return render_template('next.html',data=session["input"],err=err,max=(int)(round(max[0])),min=(int)(round(min[0])),imin=session["input"],rmin=session["input"])

from flask_mail import Mail, Message

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'dellsales07@gmail.com'
app.config['MAIL_PASSWORD'] = 'DellSales@07'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)
@app.route("/forecast/mail",methods=['GET', 'POST'])
def mails():
    if request.method == 'POST':
        sub = request.form['sub']
        rec = request.form['rec1']
        msg = Message(sub, sender = 'dellsales07@gmail.com', recipients = [rec])
        print(rec)
        msg.body = "The Forecasted Sale for the specified product would lie between "+(str)(session['Minimum'])+" to "+(str)(session['Maximum'])+" Units/Year."
        mail.send(msg)
    else:
        print("Mail not sent!")   


    return redirect(url_for('index'))

if __name__ == '__main__':
   app.secret_key = 'dell'
   app.run(debug = True)
