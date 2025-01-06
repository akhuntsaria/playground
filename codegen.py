# Write function for Fibonacci numbers and a unit test for it
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5
    assert fibonacci(6) == 8
    assert fibonacci(7) == 13
    assert fibonacci(8) == 21
    assert fibonacci(9) == 34
    assert fibonacci(10) == 55
    assert fibonacci(11) == 89
    assert fibonacci(12) == 144
    assert fibonacci(13) == 233
    assert fibonacci(14) == 377
    assert fibonacci(15) == 610
    assert fibonacci(16) == 987
    assert fibonacci(17) == 1597
    assert fibonacci(18) == 2584
    assert fibonacci(19) == 4181
    assert fibonacci(20) == 6765
    assert fibonacci(21) == 10946
    assert fibonacci(22) == 17711
    assert fibonacci(23) == 28657

test_fibonacci()