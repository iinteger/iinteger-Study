# The UNIX Time-Sharing System

**Dennis M.Ritchie and Ken Tompson. Bell Laboratories**

- #### Contribution
  
  - 대화형 운영체제를 구축하고 운영하는 데, 많은 인력이나 비용이 필요하지 않다는것을 증명함

- #### Introduction
  
  * UNIX : 범용 다중 사용자 대화형 운영체제
  
  * 하드웨어 구축에 40,000 $ 이하, 운영체제 구축에 2년 이하로 소모
  
  * 기존의 거대한 운영체제에서 지원하지 않는 다양한 기능이 탑재
    
    * ex) 텍스트 에디터, 링크 로더, 컴파일러 ...
  - UNIX의 모든 기능은 로컬에서 작성되었고, C로 짜여져 유지보수에 용이

- #### The File System
  
  - UNIX의 가장 큰 역할 : 파일시스템을 제공하는 것
  
  - 사용자 관점에서 크게 Ordinary disk file, Directory, Special file로 나뉨
    
    1. Ordinary disk file
       
       * 디스크에 저장되는 일반적인 파일
       
       * 텍스트 파일, 바이너리 파일, 사용자 정의 프로그램 등
    
    2. Directory
       
       * 파일 간 계층 구조를 정의
       
       * 모든 디렉토리의 꼭대기에는 루트 디렉토리가 존재하고, 시스템 상의 모든 파일은 이 루트 디렉토리를 추적해서 찾을 수 있음
       
       * 파일은 14자 이하의 단어로 명시되고
    
    3. Pooling Layer를 Strided Convolution(D)과 fractional-strided Convolution(G)로 대체. fractional-strided convolution이란 Transposed Convolution을 말함

    2. 두 모델에 BatchNormalization 사용. 이때 모든 Layer에 사용하는것은 아니고, G의 Output Layer와 D의 Input Layer에도 사용하지 않음  
    -> Gradient smoothing이 Mode Collapsing을 완화해주지만 학습의 대상이 되는 Original Image의 변질은 Generation task에 좋지 못한 영향을 끼치기 때문이 아닐까 예상됨
    
    
    3. Fully Connected Layer 제거
    
    
    4. Generator의 모든 활성화함수로 ReLU 사용. 이때 Output만 Tanh 사용
    
    
    5. Discriminator의 모든 활성화함수로 LeakyReLU 사용  

- **Empirical research, testing(노가다)의 결과물이기 때문에 명확한 이유를 알기 어려움**

- #### Experiment (중요)
  
  - LSUN
    
    - Generator의 data memorizing(overfitting)을 막기 위해 서로 유사한 형태를 띄는 데이터를 모두 삭제하고 실험을 진행함. 약 275000개의 데이터를 삭제
    
    - 순서대로 1 epoch 학습한 후의 결과와 5 epoch 학습한 후의 결과. Overfitting이 일어나기엔 아주 낮은 학습 횟수임에도 불구하고 생성된 데이터들의 퀄리티가 상당히 높음. 심지어 두번째 결과는 아직 underfitting된 상태라고 말함. memorizing이 발생하지 않았음을 증명
    
    - 9개의 random vector를 보간(interpolation)하며 이미지를 생성한 결과. 벡터에서 벡터로 넘어갈때, 이미지도 자연스럽게 보간되는것을 확인할 수 있음. 특히, 6번째 줄에서 점차 창문이 생기는 부분이나, 10번째 줄에서 TV가 창문으로 변화하는 부분은 굉장히 자연스러움. 논문에서는 이를 Walking in the Latent space라고 표현
    
    - 오른쪽의 Trained Filter Visualize 결과, 각 필터가 침대나 창문 등, 어느정도 특정한 Object를 담당해 학습했다는 것을 알 수 있음. 반대로 왼쪽의 Random filter는 단순히 이미지를 그대로 memorizing함.
    
    - 윗쪽은 un-modified model output, 아랫쪽은 "window"를 담당하는 filter를 dropout시킨 후 생성한 output. window가 위치하던 자리엔 어색하지만 주변과 유사한 형태로 대체됨. 모델이 아주 잘 학습되었고, 반대로 의도적으로 망각시키는 것도 가능한것으로 볼 수 있을듯

- Faces
  
  - 이미지에 대한 벡터 산술 연산 결과. 연산의 operand image는 각각의 class에 해당하는 이미지 3개의 mean vector를 사용해서 연산함. 아래쪽의 단순 픽셀 연산 결과에 비해 상대적으로 자연스러운 결과가 생성된 것을 볼 수 있음.
    
    - 왼쪽을 쳐다보는 얼굴에서 오른쪽을 쳐다보는 얼굴로 보간을 사용해서 "Turn" 한 결과.

- 위 실험 결과들은 모델이 latent vector를 output image와 단순히 1:1 mapping을 하는 것이 아닌, 데이터를 **이해**하고 있다는 것을 뒷받침함

- 그러나 training이 길어질 경우, mode collapsing이나 oscillating mode 같은 문제가 간헐적으로 발생하기 때문에 이를 해결하여야 하며, Latent Space의 property를 파악하기 위한 노력을 후속 연구로
