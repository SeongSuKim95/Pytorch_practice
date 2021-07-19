import argparse

# 인자값을 받을 instance 생성
parser = argparse.ArgumentParser(description = '사용법 테스트 입니다.')

# 입력 받을 인자값 등록
parser.add_argument('--target', required=True , help = '어느 것을 요구하는지')
parser.add_argument('--env', required=False, default = 'dev', help = '실행환경은 뭐냐')

# 입력받은 인자값을 args에 저장

args = parser.parse_args()

#입력받은 인자값 출력

print(args.target)
print(args.env)