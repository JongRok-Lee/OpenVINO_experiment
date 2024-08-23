# 프로젝트 주제
데이터 타입과 내장 그래픽 카드 사용 유무에 따른 추론 시간 분석

# 방법
OpenVINO 프레임워크 상, YOLOv10 모델을 FP32, FP16, INT8로 CPU에서 GPU에서 동일한 조건으로 추론해 추론에 소요되는 시간을 측정한다.
- 사용 모델: YOLOv10n (FLOPs: 6.7G)
- CPU: I5-1135G7, 153.6 GFLOPS
- 내장 그래픽: Iris(R) Xe Graphics, (FP32: 1,408 GFLOPS, FP16: 2,816 GFLOPS)
- 측정 방법: OpenVINOㅇ서 제공하는 benchmark_app 사용, 직접 time 라이브러 활용하여 계산
  - ```bash
    $ benchmark_app -m models/FP32_openvino_model/yolov10n.xml -d CPU -api async -shape "[1,3,640,640]" -t 15 -infer_precision f32
    $ benchmark_app -m models/FP16_openvino_model/yolov10n.xml -d CPU -api async -shape "[1,3,640,640]" -t 15 -infer_precision
    $ benchmark_app -m models/INT8_openvino_model/yolov10n.xml -d CPU -api async -shape "[1,3,640,640]" -t 15 -infer_precision
    $ python main.py
    ```
# 결과
- [1,3,640,640] 이미지 기준

- Input data: (mp4 영상)
  - GPU
    - FP32: 27.3 ms, 36.6 FPS
    - FP16: 27.1 ms, 36.8 FPS
    - INT8: 18.1 ms, 55.4 FPS
  - CPU
    - FP32: 53.1 ms, 18.8 FPS
    - FP16: 52.7 ms, 19.0 FPS
    - INT8: 26.0 ms, 38.5 FPS
- benchmark_app (더미 인풋)
  - GPU
    - FP32: 48.71 FPS
    - FP16: 84.31 FPS
    - INT8 135.52 FPS
  - CPU:
    - FP32: 38.74 FPS
    - FP16: 38.82 FPS
    - INT8: 87.38 FPS

# 결론
1. OpenVINO는 자체적으로 FP32 모델도 FP16으로 내부적으로 추론함.
2. CPU는 FP16과 FP32 추론 시간에 영향이 없으므로 둘 중에서는 조금 더 정확한 FP32 사용
3. 내장 GPU 사용 시, 추론 속도가 크게 증가하며, 특히 INT8 타입의 추론 속도 많이 향상됨

# 출처
- iGPU 스펙: https://www.techpowerup.com/gpu-specs/iris-xe-graphics-g7-80eu-mobile.c3678
- iGPU FP16 가속화 근거: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/intel-iris-xe-gpu-architecture.html
- CPU 스펙: https://www.intel.co.kr/content/www/kr/ko/support/articles/000005755/processors.html
- FP32와 FP16이 다르지 않은 이유: https://www.intel.com/content/www/us/en/support/articles/000095716/software/development-software.html
