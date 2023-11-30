#include <fstream>

#include "armnn/IRuntime.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Types.hpp"
#include "armnnDeserializer/IDeserializer.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#include "armnnOnnxParser/IOnnxParser.hpp"

using namespace armnn;

class Ann
{

public:
    int load(const char *modelPath, const char *inputName, const char *outputName, bool fastMath, bool fp16, bool saveCachedNetwork, const char *cachedNetworkPath)
    {
        BindingPointInfo inputInfo;
        BindingPointInfo outputInfo;
        INetworkPtr network = loadModel(modelPath, inputName, outputName, inputInfo, outputInfo);

        auto n = network.get();

        IOptimizedNetworkPtr optNet = OptimizeNetwork(n, fastMath, fp16, saveCachedNetwork, cachedNetworkPath);
        NetworkId netId;
        Status status = runtime->LoadNetwork(netId, std::move(optNet));
        inputInfos[netId] = inputInfo;
        outputInfos[netId] = outputInfo;
        return netId;
    }

    void embed(NetworkId netId, const void *inputData, void *outputData)
    {
        const BindingPointInfo *inputInfo = &inputInfos[netId];
        const BindingPointInfo *outputInfo = &outputInfos[netId];
        InputTensors inputTensors = {{inputInfo->first, ConstTensor{inputInfo->second, inputData}}};
        OutputTensors outputTensors = {{outputInfo->first, armnn::Tensor{outputInfo->second, outputData}}};
        runtime->EnqueueWorkload(netId, inputTensors, outputTensors);
    }

    void unload(NetworkId netId)
    {
        runtime->UnloadNetwork(netId);
    }

    unsigned long shape(NetworkId netId, bool isInput)
    {
        const TensorShape shape = (isInput ? inputInfos : outputInfos)[netId].second.GetShape();
        unsigned long s = 0;
        for (unsigned int d = 0; d < shape.GetNumDimensions(); d++)
            s |= ((unsigned long)shape[d]) << (d * 16); // stores up to 4 16-bit values in a 64-bit value
        return s;
    }

    Ann(int tuningLevel, const char *tuningFile)
    {
        IRuntime::CreationOptions runtimeOptions;
        BackendOptions backendOptions{"GpuAcc",
                                      {
                                          {"TuningLevel", tuningLevel},
                                          {"MemoryOptimizerStrategy", "ConstantMemoryStrategy"}, // SingleAxisPriorityList or ConstantMemoryStrategy
                                      }};
        if (tuningFile)
            backendOptions.AddOption({"TuningFile", tuningFile});
        runtimeOptions.m_BackendOptions.emplace_back(backendOptions);
        runtime = IRuntime::CreateRaw(runtimeOptions);
    };
    ~Ann()
    {
        IRuntime::Destroy(runtime);
    };

private:
    INetworkPtr loadModel(const char *modelPath, const char *inputName, const char *outputName, BindingPointInfo &inputInfo, BindingPointInfo &outputInfo)
    {
        const auto path = std::string(modelPath);
        if (path.rfind(".tflite") == path.length() - 7) // endsWith()
        {
            auto parser = armnnTfLiteParser::ITfLiteParser::CreateRaw();
            INetworkPtr network = parser->CreateNetworkFromBinaryFile(modelPath);
            auto inputBinding = parser->GetNetworkInputBindingInfo(0, inputName);
            inputInfo = getInputTensorInfo(inputBinding.first, inputBinding.second);
            outputInfo = parser->GetNetworkOutputBindingInfo(0, outputName);
            return network;
        }
        else if (path.rfind(".onnx") == path.length() - 5) // endsWith()
        {
            auto parser = armnnOnnxParser::IOnnxParser::CreateRaw();
            INetworkPtr network = parser->CreateNetworkFromBinaryFile(modelPath);
            auto inputBinding = parser->GetNetworkInputBindingInfo(inputName);
            inputInfo = getInputTensorInfo(inputBinding.first, inputBinding.second);
            outputInfo = parser->GetNetworkOutputBindingInfo(outputName);
            return network;
        }
        else
        {
            std::ifstream ifs(path, std::ifstream::in | std::ifstream::binary);
            auto parser = armnnDeserializer::IDeserializer::CreateRaw();
            INetworkPtr network = parser->CreateNetworkFromBinary(ifs);
            auto inputBinding = parser->GetNetworkInputBindingInfo(0, inputName);
            inputInfo = getInputTensorInfo(inputBinding.m_BindingId, inputBinding.m_TensorInfo);
            auto outputBinding = parser->GetNetworkOutputBindingInfo(0, outputName);
            outputInfo = {outputBinding.m_BindingId, outputBinding.m_TensorInfo};
            return network;
        }
    }

    BindingPointInfo getInputTensorInfo(LayerBindingId inputBindingId, TensorInfo &info)
    {
        const auto newInfo = TensorInfo{info.GetShape(), info.GetDataType(),
                                        info.GetQuantizationScale(),
                                        info.GetQuantizationOffset(),
                                        true};
        return {inputBindingId, newInfo};
    }

    IOptimizedNetworkPtr OptimizeNetwork(INetwork *network, bool fastMath, bool fp16, bool saveCachedNetwork, const char *cachedNetworkPath)
    {
        const bool allowExpandedDims = false;
        const ShapeInferenceMethod shapeInferenceMethod = ShapeInferenceMethod::ValidateOnly;

        OptimizerOptionsOpaque options;
        options.SetReduceFp32ToFp16(fp16);
        options.SetShapeInferenceMethod(shapeInferenceMethod);
        options.SetAllowExpandedDims(allowExpandedDims);

        BackendOptions gpuAcc("GpuAcc", {{"FastMathEnabled", fastMath}});
        if (cachedNetworkPath)
        {
            gpuAcc.AddOption({"SaveCachedNetwork", saveCachedNetwork});
            gpuAcc.AddOption({"CachedNetworkFilePath", cachedNetworkPath});
        }
        options.AddModelOption(gpuAcc);

        // No point in using ARMNN for CPU, use ONNX instead.
        // BackendOptions cpuAcc("CpuAcc",
        //                       {
        //                           {"FastMathEnabled", fastMath},
        //                           {"NumberOfThreads", 0},
        //                       });
        // options.AddModelOption(cpuAcc);

        BackendOptions allowExDimOpt("AllowExpandedDims",
                                     {{"AllowExpandedDims", allowExpandedDims}});
        options.AddModelOption(allowExDimOpt);
        BackendOptions shapeInferOpt("ShapeInferenceMethod",
                                     {{"InferAndValidate", shapeInferenceMethod == ShapeInferenceMethod::InferAndValidate}});
        options.AddModelOption(shapeInferOpt);

        std::vector<BackendId> backends = {
            BackendId("GpuAcc"),
            // BackendId("CpuAcc"),
            // BackendId("CpuRef"),
        };
        return Optimize(*network, backends, runtime->GetDeviceSpec(), options);
    }
    IRuntime *runtime;
    std::map<NetworkId, BindingPointInfo> inputInfos;
    std::map<NetworkId, BindingPointInfo> outputInfos;
};

extern "C" void *init(int logLevel, int tuningLevel, const char *tuningFile)
{
    LogSeverity level = static_cast<LogSeverity>(logLevel);
    ConfigureLogging(true, true, level);

    Ann *ann = new Ann(tuningLevel, tuningFile);
    return ann;
}

extern "C" void destroy(void *ann)
{
    delete ((Ann *)ann);
}

extern "C" int load(void *ann,
                    const char *path,
                    const char *inputName,
                    const char *ouputName,
                    bool fastMath,
                    bool fp16,
                    bool saveCachedNetwork,
                    const char *cachedNetworkPath)
{
    return ((Ann *)ann)->load(path, inputName, ouputName, fastMath, fp16, saveCachedNetwork, cachedNetworkPath);
}

extern "C" void unload(void *ann, NetworkId netId)
{
    ((Ann *)ann)->unload(netId);
}

extern "C" void embed(void *ann, NetworkId netId, void *inputData, void *outputData)
{
    ((Ann *)ann)->embed(netId, inputData, outputData);
}

extern "C" unsigned long shape(void *ann, NetworkId netId, bool isInput)
{
    return ((Ann *)ann)->shape(netId, isInput);
}