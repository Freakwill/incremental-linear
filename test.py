
from incremental_linear import *

import pytest

class Test_IL:
    def test_scalar_matrix(self):
        print('With Chinese:')
        show_gender("蔡徐坤")
        assert np.all(scalar_matrix(3, 1)==np.eye(3))
    def test_pinyin(self):
        print('With Pinyin:')
        show_gender("蔡徐坤", True)
        assert True
    def test_find(self):
        k = find('cherry', ['apple', 'banana', 'cherry', 'peach'])
        assert k == 2
        k = find('pair', ['apple', 'banana', 'cherry', 'peach'])
        assert k == -1
        
if __name__ == '__main__':
    pytest.main("-s test_il.py")
