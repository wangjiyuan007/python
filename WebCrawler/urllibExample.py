import urllib.request
import urllib.parse

#response = urllib.request.urlopen('https://www.python.org')
#print(response.read().decode('utf-8'))
#print(type(response))
#print(response.status)
#print(response.getheaders())
#print(response.getheader('Server'))
'''data = bytes(urllib.parse.urlencode({'word':'hello'}),encoding='utf-8')
response = urllib.request.urlopen('http://httpbin.org/post',data=data)'''
response = urllib.request.urlopen('http://httpbin.org/get',timeout=1)
print(response.read())