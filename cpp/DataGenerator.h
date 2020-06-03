/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: May 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef DATAGENERATOR_H
#define DATAGENERATOR_H 1

#include "ecvl/support_eddl.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <iostream>

using namespace eddl;
using namespace ecvl;
using namespace std;

/*
    This is a class in development, it is pending to decide where
    it should be included within the code.
*/

class DataGenerator
{

    class my_pair
    {
        public:
            my_pair( tensor x, tensor y ) : x(x), y(y)
            {
            }

            tensor x;
            tensor y;
    };

public:
    DataGenerator( ecvl::DLDataset * source, int  batch_size, std::vector<int>   input_shape, std::vector<int>   output_shape, int n_producers=5 )
        : source(source),
          batch_size(batch_size),
          input_shape(input_shape),
          output_shape(output_shape),
          n_producers(n_producers)
    {
        num_samples = vsize(source->GetSplit());
        num_batches = num_samples / batch_size;

        producers = new std::thread [n_producers];

        active = false;
        batch_index = 0;
    }
    ~DataGenerator()
    {
        if ( active ) stop();

        // this loop can be executed with no control of mutex if producer stopped
        while( ! fifo.empty() ) {

            my_pair * _temp = fifo.front(); fifo.pop();

            delete _temp->x;
            delete _temp->y;
            delete _temp;
        }

        delete [] producers;
    }

    void start()
    {
        if ( active ) {
            cerr << "FATAL ERROR: trying to start the producer threads when they are already running!" << endl;
            abort();
            // msg("FATAL ERROR: trying to start the producer threads when they are already running!");
        }
        batch_index=0;
        active = true;
        for( int i=0; i < n_producers; i++ )
            producers[i] = std::thread( & DataGenerator::thread_producer, this );
    }
    void stop()
    {
        if ( ! active ) {
            cerr << "FATAL ERROR: trying to stop the producer threads when they are stopped!" << endl;
            abort();
            // msg("FATAL ERROR: trying to stop the producer threads when they are stopped!");
        }

        active = false;
        for( int i=0; i < n_producers; i++ ) producers[i].join();
    }

    void thread_producer()
    {
        std::queue<my_pair *>   my_cache;

        while( active  &&  batch_index < num_batches ) {

            int j=-1;

            { // critical region starts
                std::unique_lock<std::mutex> lck(mutex_batch_index);

                j=batch_index++;
            } // critical region ends


            if ( j >= num_batches ) break;

            // creating new tensors for every batch can generate overload, let us check now this in the future
            tensor x = new Tensor({ batch_size, source->n_channels_, input_shape[0], input_shape[1] });
            tensor y = new Tensor({ batch_size, static_cast<int>(source->classes_.size()) });

            // Load a batch
            source->LoadBatch( x, y );

            my_pair * p = new my_pair(x,y);

            my_cache.push( p );

            if ( fifo.size() < 20  ||  my_cache.size() > 5 ) {
                { // critical region starts
                    std::unique_lock<std::mutex> lck(mutex_fifo);

                    while( ! my_cache.empty() ) {
                        fifo.push( my_cache.front() ); my_cache.pop();
                    }
                } // critical region ends
                cond_var_fifo.notify_one();
            }

            // pending to be adjusted
            while( fifo.size() > 100 ) std::this_thread::sleep_for( std::chrono::seconds(1) );
        }

        if ( my_cache.size() > 0 ) {
            { // critical region starts
                std::unique_lock<std::mutex> lck(mutex_fifo);
                while( ! my_cache.empty() ) {
                    fifo.push( my_cache.front() ); my_cache.pop();
                }
            } // critical region ends
            cond_var_fifo.notify_one();
        }
    }

    bool has_next()
    {
        return batch_index < num_batches  ||  fifo.size() > 0;
    }

    int size()
    {
        return fifo.size();
    }

    bool pop_batch( tensor & x, tensor & y )
    {
        my_pair * _temp = nullptr;
        { // critical region begins
            std::unique_lock<std::mutex> lck(mutex_fifo);

            if ( fifo.empty() ) cond_var_fifo.wait( lck );
            if ( fifo.empty() ) return false;

            _temp = fifo.front(); fifo.pop();
        } // critical region ends

        x = _temp->x;
        y = _temp->y;

        delete _temp;

        return true;
    }

private:
    ecvl::DLDataset   * source;
    int                 batch_size;
    std::vector<int>    input_shape;
    std::vector<int>    output_shape;
    int                 n_producers;

    std::queue<my_pair *>     fifo;
    std::mutex                mutex_fifo;
    std::mutex                mutex_batch_index;
    std::condition_variable   cond_var_fifo;
    std::thread             * producers;
    int                       batch_index;

    bool    active;
    int     num_samples;
    int     num_batches;
};

#endif // DATAGENERATOR_H
