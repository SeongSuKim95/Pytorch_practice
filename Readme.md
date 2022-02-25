# Repository for practice some pytorch skills


## 1. Error message
-------------------


## 2. Concept
  - model.named_modules() [[LINK]](https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8) : 자신을 포함한 모든 submodule을 반환
  - hook [[LINK]](https://velog.io/@bluegun/Pytorch%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%EA%B5%AC%EC%84%B1-%ED%95%99%EC%8A%B5-%EB%93%B1-%EC%A0%95%EB%A6%AC) 
    - Forward 나 Backward 전후로,  사용자가 원하는 custom function을 삽입하는 기능
    - module 은 forward, backward hook 전부 가능
    - tensor는 backward hook만 가능
    
        ```python
        tensor.register_hook(custom function)
        model.register_forward_pre_hook(...) # forward 실행 전에 삽입
        model.register_forward_hook(...) # forward 실행 후에 삽입
        model.register_full_backward_hook(...) # module input에 대한 gradient가 계산될 때마다 호출
        ```
 - Apply : model 전체에 custom function을 적용하고 싶을 떄 사용
    ```python
    ret = model.apply(custom function) # custom function을 model 전체에 적용. apply가 적용된 model을 return
    ```


  

