#include "vector_compression.hpp"

#include <map>
#include <memory>

#include "utils/assert.hpp"

#include "bitpacking/bitpacking_compressor.hpp"
#include "fixed_width_integer/fixed_width_integer_compressor.hpp"

namespace hyrise {

namespace {

/**
 * @brief Mapping of vector compression types to compressors
 *
 * Add your vector compressor here!
 */
const auto vector_compressor_for_type = std::map<VectorCompressionType, std::shared_ptr<AbstractVectorCompressor>>{
    {VectorCompressionType::FixedWidthInteger, std::make_shared<FixedWidthIntegerCompressor>()},
    {VectorCompressionType::BitPacking, std::make_shared<BitPackingCompressor>()}};

std::unique_ptr<AbstractVectorCompressor> create_compressor_by_type(VectorCompressionType type) {
  auto it = vector_compressor_for_type.find(type);
  Assert(it != vector_compressor_for_type.cend(),
         "All vector compression types must be in vector_compressor_for_type.");

  const auto& compressor = it->second;
  return compressor->create_new();
}

}  // namespace

std::unique_ptr<const AbstractCompressedVector> compress_vector(const pmr_vector<uint32_t>& vector,
                                                            const VectorCompressionType type,
                                                            const PolymorphicAllocator<size_t>& alloc,
                                                            const UncompressedVectorInfo& meta_info) {
  auto compressor = create_compressor_by_type(type);
  return compressor->compress(vector, alloc, meta_info);
}

}  // namespace hyrise
