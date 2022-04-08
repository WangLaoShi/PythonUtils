import requests
import json
import time
import random


def getToutiaoFeed(max_behot_time = 0):
    url = "https://www.toutiao.com/api/pc/list/user/feed?category={}&token={}&max_behot_time={}&aid=24&app_name={}&_signature={}"

    category = 'my_favorites'
    token = '********************'
    app_name = 'toutiao_web'
    signature = '*****************'

    url = url.format(category, token, max_behot_time, app_name, signature)

    # print(url)

    payload = {}
    headers = {
        'authority': 'www.toutiao.com',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
        'accept': 'application/json, text/plain, */*',
        'dnt': '1',
        'sec-ch-ua-mobile': '?0',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': '************************',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'cookie': '**********************'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    content = (response.text)

    jsonData = json.loads(content)
    # print(jsonData)
    if jsonData and jsonData['message'] == 'success':

        for articele in jsonData['data']:
            print('<tr>')
            output = ''
            if articele.get('title'):
                output += '<td>'+articele['title'] + '</td>'
            else:
                output += '<td>'+'</td>'

            if articele.get('abstract'):
                output += '<td>'+articele['abstract'] + '</td>'
            else:
                output += '<td>'+'</td>'

            if articele.get('behot_time'):
                output += '<td>'+str(articele['behot_time']) + '</td>'
            else:
                output += '<td>'+'</td>'

            if articele.get('article_url'):
                output += '<td>'+str(articele['article_url']) + '</td>'
            else:
                output += '<td>'+'</td>'

            if articele.get('display_url'):
                output += '<td>'+str(articele['display_url']) + '</td>'
            else:
                output += '<td>'+'</td>'

            if articele.get('share_url'):
                output += '<td>'+str(articele['share_url']) + '</td>'
            else:
                output += '<td>'+'</td>'

            if articele.get('source'):
                output += '<td>'+str(articele['source']) + '</td>'
            else:
                output += '<td>'+'</td>'

            if articele.get('rich_content'):
                output += '<td>'+str(articele['rich_content']) + '</td>'
            else:
                output += '<td>'+'</td>'

            print(output)
            print('</tr>')
    else:
        print("数据解析出错")
        exit()

    next_behot_time = jsonData['next']['max_behot_time']

    secondRandom = random.randint(10,15)

    # print("Sleep---" + str(secondRandom))

    time.sleep(secondRandom)

    getToutiaoFeed(next_behot_time)

getToutiaoFeed(0)



