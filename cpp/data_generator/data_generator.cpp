#include "data_generator.h"

#include <iostream>

using namespace eddl;
using namespace ecvl;
using namespace std;

DataGenerator::DataGenerator(ecvl::DLDataset* source, int  batch_size, std::vector<int> input_shape, std::vector<int> output_shape, int n_producers)
    : source_(source),
    batch_size_(batch_size),
    input_shape_(input_shape),
    output_shape_(output_shape),
    n_producers_(n_producers)
{
    int num_samples = vsize(source_->GetSplit());
    num_batches_ = num_samples / batch_size_;

    producers_ = new std::thread[n_producers_];

    active_ = false;
    batch_index_ = 0;
}

DataGenerator::~DataGenerator()
{
    if (active_) {
        Stop();
    }

    // this loop can be executed with no control of mutex if producer stopped
    while (!fifo_.empty()) {
        TensorPair* _temp = fifo_.front();
        fifo_.pop();

        delete _temp->x_;
        delete _temp->y_;
        delete _temp;
    }

    delete[] producers_;
}

void DataGenerator::Start()
{
    if (active_) {
        cerr << "FATAL ERROR: trying to start the producer threads when they are already running!" << endl;
        abort();
    }

    batch_index_ = 0;
    active_ = true;
    for (int i = 0; i < n_producers_; i++) {
        producers_[i] = std::thread(&DataGenerator::ThreadProducer, this);
    }
}

void DataGenerator::Stop()
{
    if (!active_) {
        cerr << "FATAL ERROR: trying to stop the producer threads when they are stopped!" << endl;
        abort();
    }

    active_ = false;
    for (int i = 0; i < n_producers_; i++) {
        producers_[i].join();
    }
}

void DataGenerator::ThreadProducer()
{
    std::queue<TensorPair*> my_cache;

    while (active_ && batch_index_ < num_batches_) {
        int j = -1;

        { // critical region starts
            std::unique_lock<std::mutex> lck(mutex_batch_index_);

            j = batch_index_++;
        } // critical region ends

        if (j >= num_batches_) {
            break;
        }

        // creating new tensors for every batch can generate overload, let us check now this in the future
        tensor x = new Tensor({ batch_size_, source_->n_channels_, input_shape_[0], input_shape_[1] });
        tensor y;
        if(source_->classes_.empty()) {
            y = new Tensor({ batch_size_, source_->n_channels_gt_, output_shape_[0], output_shape_[1] });
        }
        else {
            y = new Tensor({ batch_size_, vsize(source_->classes_) });
        }

        // Load a batch
        source_->LoadBatch(x, y);

        TensorPair* p = new TensorPair(x, y);

        my_cache.push(p);

        if (fifo_.size() < 20 || my_cache.size() > 5) {
            { // critical region starts
                std::unique_lock<std::mutex> lck(mutex_fifo_);

                while (!my_cache.empty()) {
                    fifo_.push(my_cache.front());
                    my_cache.pop();
                }
            } // critical region ends
            cond_var_fifo_.notify_one();
        }

        // pending to be adjusted
        while (fifo_.size() > 100) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    if (my_cache.size() > 0) {
        { // critical region starts
            std::unique_lock<std::mutex> lck(mutex_fifo_);
            while (!my_cache.empty()) {
                fifo_.push(my_cache.front());
                my_cache.pop();
            }
        } // critical region ends
        cond_var_fifo_.notify_one();
    }
}

bool DataGenerator::HasNext()
{
    return batch_index_ < num_batches_ || fifo_.size() > 0;
}

size_t DataGenerator::Size()
{
    return fifo_.size();
}

bool DataGenerator::PopBatch(tensor& x, tensor& y)
{
    TensorPair* _temp = nullptr;
    { // critical region begins
        std::unique_lock<std::mutex> lck(mutex_fifo_);

        if (fifo_.empty()) {
            cond_var_fifo_.wait(lck);
        }

        if (fifo_.empty()) {
            return false;
        }

        _temp = fifo_.front();
        fifo_.pop();
    } // critical region ends

    x = _temp->x_;
    y = _temp->y_;

    delete _temp;

    return true;
}