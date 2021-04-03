#pragma once

#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore
{
namespace dataset
{
class ElasticSampler
{
  public:
    // Constructor
    // @param int64_t num_samples: the user-requested number of samples ids to
    // generate. A value of 0
    //                indicates that the sampler should produce the complete set
    //                of ids.
    // @param int64_t samplesPerBuffer: Num of Sampler Ids to fetch via 1
    // GetNextBuffer call
    ElasticSampler(int64_t num_samples, int64_t samples_per_buffer);

    ElasticSampler(const ElasticSampler &s)
        : ElasticSampler(s.num_samples_, s.samples_per_buffer_)
    {
    }

    // default destructor
    ~ElasticSampler() = default;

    // Get a list of sample ids.
    // @note It is Sampler responsibility to make sure that the id is not out of
    // bound.
    // @param std::unique_ptr<DataBuffer> pBuffer - Buffer to be returned to
    // StorageOp
    // @param int32_t workerId - not meant to be used
    // @return Status The status code returned
    virtual Status GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) = 0;

// This function only called by python layer. Not needed by Android.
#ifdef ENABLE_PYTHON
    // return all ids in one epoch as a numpy array, then call reset
    Status GetAllIdsThenReset(py::array *data);
#endif

    // for next epoch of sampleIds
    // @return Status The status code returned
    virtual Status ResetSampler() = 0;

    // first handshake between leaf source op and Sampler. This func will
    // determine the amount of data in the dataset that we can sample from.
    // @param op - leaf op pointer, pass in so Sampler can ask it about how much
    // data there is
    // @return
    virtual Status HandshakeRandomAccessOp(const RandomAccessOp *op);

    // initialize sampler and perform checks on certain vars
    virtual Status InitSampler()
    {
        return Status::OK();
    }

    // setter for num samples
    // @param num_samples - the number of samples to assign.
    // @return status error code
    Status SetNumSamples(int64_t num_samples);

    // getter for num samples
    // @return number of samples
    int64_t GetNumSamples();

    // Calculate num samples. Unlike GetNumSamples, it is not a getter and
    // doesn't necessarily return the value of num_samples_
    // @return number of samples
    virtual int64_t CalculateNumSamples(int64_t num_rows);

    // setter for num or records in the dataset
    // @param num_rows - the number of records
    // @return status error code
    Status SetNumRowsInDataset(int64_t num_rows);

    // Adds a sampler to become our child.
    // @param std::shared_ptr<DatasetOp> - The sampler to add as a child.
    // @return Status The status code returned
    Status AddChild(std::shared_ptr<ElasticSampler> child);

    // A helper function to create a int64_t 1-D Tensor specifically used to
    // hold sampleIds for Sampler
    // @param std::shared_ptr<Tensor>* sampleIds
    // @param int64_t numElements - must be a non 0 number
    // @return Status The status code returned
    Status CreateSamplerTensor(std::shared_ptr<Tensor> *sample_ids,
                               int64_t num_elements);

    // A print method typically used for debugging
    // @param out - The output stream to write output to
    // @param show_all - A bool to control if you want to show all info or just
    // a summary
    virtual void SamplerPrint(std::ostream &out, bool show_all) const;

    // << Stream output operator overload
    // @notes This allows you to write the debug print info using stream
    // operators
    // @param out - reference to the output stream being overloaded
    // @param sampler - reference to teh sampler to print
    // @return - the output stream must be returned
    friend std::ostream &operator<<(std::ostream &out,
                                    const ElasticSampler &sampler)
    {
        sampler.SamplerPrint(out, false);
        return out;
    }

    // Checks if this sampler has a child sampler.
    // @return - tre if there is a child sampler, false otherwise.
    bool HasChildSampler();

    // Uses id as an index for the list of ids generated by the child sampler,
    // and gets the associated id.
    // @param int64_t* out_associated_id - Out parameter, contains the
    // associated id.
    // @param int64_t id - The id used as an index to get the associated child
    // id.
    // @return Status The status code returned
    Status GetAssociatedChildId(int64_t *out_associated_id, int64_t id);

  protected:
    // Number of rows of data from the place this sampler is sampling from. If
    // this sampler has a child sampler, num_rows_ is the number of ids the
    // child sampler will output. Otherwise, num_rows_ is the number of rows in
    // the dataset.
    int64_t num_rows_;

    // The user may want to sample less than the full amount of data.
    // num_samples_ reduces the number of id's returned as request by the user.
    // Derived classes will choose how to sample the smaller amount.
    int64_t num_samples_;

    int64_t samples_per_buffer_;
    std::unique_ptr<ColDescriptor> col_desc_;
    std::vector<std::shared_ptr<ElasticSampler>> child_;  // Child nodes
    std::unique_ptr<DataBuffer> child_ids_;
};
}  // namespace dataset
}  // namespace mindspore
