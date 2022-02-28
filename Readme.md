# Repository for practice some pytorch skills


## 1. Error message
-------------------
  - ImportError: cannot import name 'container_abcs' from 'torch._six'
    - timm의 old version 문제에서 발생한다. 아래와 같이 container_abcs의 import name을 바꿔주면 해결됨
      ```python
      import collections.abc as container_abcs
      ```

## 2.Enviornment settings
-------------------
  - Conda enviornmnet 삭제
    ```python
    conda env remove -n "environment name"
    ```
  - RTX 3000 번대 부터는 Cudatoolkit version이 11.1 이상이어야 한다. Requirements를 깔기전에 cuda version에 맞는 torch와 torchvision을 깔아 주어야 문제가 안생긴다.
    ```python
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
    ```

## 2. Concept
  - model.named_modules() [[LINK]](https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8) : 자신을 포함한 모든 submodule을 반환
  - hook [[LINK]](https://velog.io/@bluegun/Pytorch%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%EA%B5%AC%EC%84%B1-%ED%95%99%EC%8A%B5-%EB%93%B1-%EC%A0%95%EB%A6%AC) [[LINK]](https://daebaq27.tistory.com/65) 
    - Pytorch hook Tutorial [[LINK]](https://hongl.tistory.com/157)
    - Forward 나 Backward 전후로,  사용자가 원하는 custom function을 삽입하는 기능
    - pre-hook : 프로그램 실행 전에 걸어놓는 hook
    - forward hook
      - register_forward_hook : forward 호출 후에 forward output 계산 후 걸어두는 hook
      - register_forward_pre_hook : forward 호출 전에 걸어두는 hook
    - backward hook
      - register_full_backward_hook(module에 적용) : module input에 대한 gradient가 계산될 떄마다 hook이 호출됨
      - hook(module, grad_input, grad_output) -> tuple(Tensor) or None
      - grad_input과 grad_outut은 각각 input과 output에 대한 gradient를 포함하고 있는 tuple
    - module 은 forward, backward hook 전부 가능
    - tensor는 backward hook만 가능
    
        ```python
        tensor.register_hook(custom function)
        model.register_forward_pre_hook(...) # forward 실행 전에 삽입
        model.register_forward_hook(...) # forward 실행 후에 삽입
        model.register_full_backward_hook(...) # module input에 대한 gradient가 계산될 때마다 호출
        ```
    - hook 활용 (feature extraction, visualizing activation)
      ```python
      """
      author: Frank Odom
      https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
      """
      from typing import Dict, Iterable, Callable

      class FeatureExtractor(nn.Module):
          def __init__(self, model: nn.Module, layers: Iterable[str]):
              super().__init__()
              self.model = model
              self.layers = layers
              self._features = {layer: torch.empty(0) for layer in layers}

              for layer_id in layers:
                  layer = dict([*self.model.named_modules()])[layer_id]
                  layer.register_forward_hook(self.save_outputs_hook(layer_id))

          def save_outputs_hook(self, layer_id: str) -> Callable:
              def fn(_, __, output):
                  self._features[layer_id] = output
              return fn

          def forward(self, x: Tensor) -> Dict[str, Tensor]:
              _ = self.model(x)
              return self._features​
        ```
        ```python 
        """
        author: Ayoosh Kathuria
        https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
        """
        import torch 
        import torch.nn as nn

        class myNet(nn.Module):
          def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3,10,2, stride = 2)
            self.relu = nn.ReLU()
            self.flatten = lambda x: x.view(-1)
            self.fc1 = nn.Linear(160,5)
            self.seq = nn.Sequential(nn.Linear(5,3), nn.Linear(3,2))
            
          
          
          def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.fc1(self.flatten(x))
            x = self.seq(x)
          

        net = myNet()
        visualisation = {}

        def hook_fn(m, i, o):
          visualisation[m] = o 

        def get_all_layers(net):
          for name, layer in net._modules.items():
            #If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
              get_all_layers(layer)
            else:
              # it's a non sequential. Register a hook
              layer.register_forward_hook(hook_fn)

        get_all_layers(net)

          
        out = net(torch.randn(1,3,8,8))

        # Just to check whether we got all layers
        visualisation.keys()      #output includes sequential layers​
        ```
 - Apply : model 전체에 custom function을 적용하고 싶을 떄 사용
    ```python
    ret = model.apply(custom function) # custom function을 model 전체에 적용. apply가 적용된 model을 return
    ```
 - 정규식
   - re.compile : import re 를 사용하여 re로부터 직접 메서드를 호출하는것이 반복적으로 사용해야할 경우 부담이 되기 때문에, 컴파일을 미리 해 두고 이를 저장하는 것
   - Example [[LINK]](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/)
     ```python
      print(re.search(r'\b\d+\b', 'Ryan 1 Tube 2 345'))
      # 만약 이런 정규식을 반복적으로 써야한다면?
      
      with open('ryan.txt', 'r', encoding='utf-8') as f_in:
        reObj = re.compile(r'\b\d+\b')
        for line in f_in:
            matchObj = reObj.search(line)
            print(matchObj.group())
      # 이런식으로 써주자!
      # 무슨 차이인가?
      # 지금까지는 re 모듈로부터 직접 match, search 등 메서드를 써 왔고, 인자는 기본적으로 1) 정규식 패턴과 2) 찾을 문자열이 있었다.
      
      # 컴파일을 미리 하는 버전은,

      # re.compile 메서드로부터 reObj를 만든다.
      # 인자는 기본적으로 1) 정규식 패턴 하나이다.
      # reObj.match 혹은 search 등으로 문자열을 찾는다.
      # 인자는 정규식 패턴은 이미 저장되어 있으므로 search 메서드에는 1) 찾을 문자열 하나만 주면 된다.
      
     ```
  
 - list 
   - 반복문 안에서 list.append를 아래와 같이도 구현할 수 있다.
     ```python
     for _, pid, camid, trackid in data:  # _=directory, pid=pid, camid=camid
        pids += [pid] # pid를 list에 append
        cams += [camid]
        tracks += [trackid]
     ```


  

