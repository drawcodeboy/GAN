### 1. 01_generate
```
python subtasks/01_generate/exec.py
```
* generation 해보는 예시
* <code>assets</code>에 저장시킨다.

### 2. 02_loss_curve
```
python subtasks/02_loss_curve/exec.py
```
* Discriminator와 Generator의 loss 변화 추이를 보면서 <b>Mode collapse</b>와 같은 현상이 일어나는지 모니터링
* 물론, Generation의 결과가 비슷하다면 이것도 <b>Mode collapse</b>라는 걸 의심할 수 있음

### 3. 03_grid_generate
```
python subtasks/03_grid_generate/exec.py
```
* 일반적인 생성 모델의 논문에서 시각화하듯이
* grid 형태로 생성 결과를 정렬하는 코드