from django.shortcuts import render
import requests

# Create your views here.
def search_panorama(request):
    if request.method == 'POST':
        address = request.POST.get('address')  # POST로부터 검색할 주소를 가져옴
        api_url = f'https://naveropenapi.apigw.ntruss.com/map-place/v1/search?query={address}&coordinate=127.1054327,37.3595963'
        headers = {
            'X-NCP-APIGW-API-KEY-ID': 'ssd8c8ip5h',  # 네이버 지도 API 클라이언트 ID
            'X-NCP-APIGW-API-KEY': 'nCHrWW8dNSFZwmhq7BMWdOxPfzZu9bUTDTmNRYm7'  # 네이버 지도 API 클라이언트 Secret
        }
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result['meta']['totalCount'] > 0:
                panorama_id = result['places'][0]['panorama']['id']
                panorama_url = f'https://map.naver.com/v5/panorama/{panorama_id}?c=127.1054327,37.3595963,120.0,0,0,0,dh'
                return render(request, 'panorama.html', {'panorama_url': panorama_url})
            else:
                return render(request, 'panorama.html', {'error_message': '주소에 해당하는 파노라마를 찾을 수 없습니다.'})
    return render(request, 'search.html')