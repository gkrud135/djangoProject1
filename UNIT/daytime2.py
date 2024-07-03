import requests
from datetime import datetime
import xml.etree.ElementTree as ET


def day2night(parsed_alert):

    # 기상청 API 정보
    url = 'http://apis.data.go.kr/B090041/openapi/service/RiseSetInfoService/getAreaRiseSetInfo'
    params = {
        'serviceKey': 'Rdz/+2beLY8tEg8H2Yyxf3yy3FL8GDcB5fXlChRpzv+ZGDUwFUk+IzjThZxTCDeK3xRXfIVQRy4iJgkWh/RFCA==',
        'locdate': datetime.today().strftime('%Y%m%d'),  # 오늘 날짜로 설정
        'location': '서울'
    }

    # API 호출
    response = requests.get(url, params=params)
    response_data = response.content

    # XML 파싱
    root = ET.fromstring(response_data)

    # XML 응답에서 일출과 일몰 시간 추출
    sunrise = root.findtext('.//sunrise').strip()
    sunset = root.findtext('.//sunset').strip()

    # 일출 및 일몰 시간 확인
    print(f"Sunrise: {sunrise}, Sunset: {sunset}")

    # 현재 시간
    now = datetime.now()
    time_str = parsed_alert['재난 발생 시간']
    parsed_time = datetime.strptime(time_str, "%H시 %M분") #여기에서 발생시간 가져와서 저장한다음에
    # 일출과 일몰 시간을 datetime 객체로 변환
    try:
        sunrise_time = datetime.strptime(sunrise, "%H%M").replace(year=now.year, month=now.month, day=now.day)
        sunset_time = datetime.strptime(sunset, "%H%M").replace(year=now.year, month=now.month, day=now.day)
        disaster_time = now.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0, microsecond=0) #여기서 nowㄹ를 대체

        # 현재 시간이 일출과 일몰 시간 사이인지 확인
        if sunrise_time < disaster_time < sunset_time:
            sun = 0
        else:
            sun = 1 #night
    except ValueError as e:
        print(f"Error parsing sunrise/sunset times: {e}")
        sun = 2
    return sun