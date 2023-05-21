# tt_transformer_bp

This is a basic C++ version for fine-grained path design of a tensorized fully-connected layer using visual studio 2022.

`test.cpp`: test corresponding function (can also be used as test benchmark for HLS)

`order_control.cpp`: fine-grained method to calculate gradients

`fc_bwd.cpp`: a slow method to calculate gradients

`contract.cpp`: do tensor contraction along B dimension
