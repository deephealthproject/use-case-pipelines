#ifndef DATAGENERATOR_H
#define DATAGENERATOR_H

#include "ecvl/support_eddl.h"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

/*
    This is a class in development, it is pending to decide where
    it should be included within the code.
*/

class DataGenerator
{
    class TensorPair
    {
    public:
        TensorPair(tensor x, tensor y) : x_(x), y_(y) {}

        tensor x_;
        tensor y_;
    };

private:
    ecvl::DLDataset*    source_;
    int                 batch_size_;
    std::vector<int>    input_shape_;
    std::vector<int>    output_shape_;
    int                 n_producers_;

    std::queue<TensorPair*>     fifo_;
    std::mutex                  mutex_fifo_;
    std::mutex                  mutex_batch_index_;
    std::condition_variable     cond_var_fifo_;
    std::thread*                producers_;
    int                         batch_index_;

    bool    active_;
    int     num_batches_;

public:
    DataGenerator(ecvl::DLDataset* source, int  batch_size, std::vector<int> input_shape, std::vector<int> output_shape, int n_producers = 5);
    ~DataGenerator();

    void Start();
    void Stop();
    void ThreadProducer();
    bool HasNext();
    size_t Size();
    bool PopBatch(tensor& x, tensor& y);
};

#endif // DATAGENERATOR_H
