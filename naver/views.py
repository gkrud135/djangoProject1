from django.shortcuts import render
import requests
from urllib.parse import quote
import json

# Create your views here.
def search_panorama(request):
    if request.method == 'POST':
        address = request.POST.get('address')  # POST로부터 검색할 주소를 가져옴
        result = json.dumps(search_naver_local(address))
        print(result)

        return render(request, 'panorama.html', {'address': address, 'result':result})

    else:
        return render(request, 'search.html', {'error_message': '잘못된 접근입니다.'})


def search_naver_local(keyword, start=1, display=1):
    headers = {
        'X-Naver-Client-Id': 'HuiILt2z5LBgzOuBBN2c',
        'X-Naver-Client-Secret': 'sBHmew0kHZ'
    }

    encoded_keyword = quote(keyword)
    url = f"https://openapi.naver.com/v1/search/local.json?query={encoded_keyword}&start={start}&display={display}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None