import nltk
import json
import urllib2
import urllib
import re

def getSearchResultsGoogle(searchString, number):
    resultsToFind = number
    searchTitles = []
    start = 0
    
    cx = '015484707549482425261:b5q1t9m6bpa'
    key = 'AIzaSyCUegTaNsC6224cy4VNKOSWr8isztlaqh4'
    index = 0
    try:
        while(resultsToFind > 0):

            query = urllib.urlencode({'q': searchString, 'start': ((index*10)+1)})
            print(query)
            url2 = 'https://www.googleapis.com/customsearch/v1?key=AIzaSyCUegTaNsC6224cy4VNKOSWr8isztlaqh4&cx=015484707549482425261:b5q1t9m6bpa&%s' % query
            search_response = urllib2.urlopen(url2)
            search_results = search_response.read().decode("utf8")
            results = json.loads(search_results)
            data = results['items']
            
            for result in data:
                searchTitles.append(result["formattedUrl"])
            
            resultsToFind = resultsToFind - 10

            start = index + 1
    except:
        pass

    
    return searchTitles

#bing limits to 50 results at a time, iterations is number of 50 result chunks to include
def getSearchResultsBing(searchString, number):

    iterations = number/50
    
    #for api
    key = '1Hryit2sBmTJrkkU57TdTjQJsAiOx0TaVZXxcA7KVbc'
    query = urllib.quote(searchString)
    search_type = 'Web'
    user_agent = 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; FDM; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 1.1.4322)'
    credentials = (':%s' % key).encode('base64')[:-1]
    auth = 'Basic %s' % credentials

    searchTitles = []
    
    for i in range(iterations):
        url = 'https://api.datamarket.azure.com/Data.ashx/Bing/Search/'+search_type+'?Query=%27'+query+'%27&$top=40&$format=json&$skip=' + str(i*50)
        print(url)

        
        request = urllib2.Request(url)
        request.add_header('Authorization', auth)
        request.add_header('User-Agent', user_agent)
        request_opener = urllib2.build_opener()
        response = request_opener.open(request)
        response_data = response.read()
        json_result = json.loads(response_data)
        result_list = json_result['d']['results']
        

        
        for result in result_list:
            searchTitles.append(result['Url'])

            
    return searchTitles

test = getSearchResultsBing("fish fry if", 50)
