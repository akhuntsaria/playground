```sh
nvcc -arch=sm_89 -o heat *.cu `pkg-config --cflags --libs opencv4` -diag-suppress 611 && ./heat
```

