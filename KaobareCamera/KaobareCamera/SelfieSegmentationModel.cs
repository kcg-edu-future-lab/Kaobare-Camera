using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace KaobareCamera;

public sealed class SelfieSegmentationModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly int _inputWidth;
    private readonly int _inputHeight;
    private readonly bool _channelsLast;

    public SelfieSegmentationModel(string modelPath)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();

        var dims = _session.InputMetadata[_inputName].Dimensions.Select(d => d <= 0 ? -1 : d).ToArray();
        // 想定: NCHW = [1,3,256,256] または NHWC = [1,256,256,3]
        if (dims.Length == 4 && dims[1] == 3)
        {
            _channelsLast = false;
            _inputHeight = dims[2] > 0 ? dims[2] : 256;
            _inputWidth = dims[3] > 0 ? dims[3] : 256;
        }
        else if (dims.Length == 4 && dims[3] == 3)
        {
            _channelsLast = true;
            _inputHeight = dims[1] > 0 ? dims[1] : 256;
            _inputWidth = dims[2] > 0 ? dims[2] : 256;
        }
        else
        {
            // Hugging Face の onnx-community/mediapipe_selfie_segmentation など、
            // 動的 shape の場合は 256x256 NCHW として扱います。
            _channelsLast = false;
            _inputWidth = 256;
            _inputHeight = 256;
        }
    }

    public Mat RemoveBackground(Mat bgrFrame, double threshold)
    {
        using var maskSmall = PredictMask(bgrFrame);
        using var mask = new Mat();
        Cv2.Resize(maskSmall, mask, new Size(bgrFrame.Width, bgrFrame.Height), 0, 0, InterpolationFlags.Linear);

        // 境界のちらつき・ギザギザを少し抑える
        Cv2.GaussianBlur(mask, mask, new Size(7, 7), 0);

        var bgra = new Mat(bgrFrame.Rows, bgrFrame.Cols, MatType.CV_8UC4);
        unsafe
        {
            for (int y = 0; y < bgrFrame.Rows; y++)
            {
                byte* src = (byte*)bgrFrame.Ptr(y).ToPointer();
                byte* m = (byte*)mask.Ptr(y).ToPointer();
                byte* dst = (byte*)bgra.Ptr(y).ToPointer();

                for (int x = 0; x < bgrFrame.Cols; x++)
                {
                    byte b = src[x * 3 + 0];
                    byte g = src[x * 3 + 1];
                    byte r = src[x * 3 + 2];
                    byte a = m[x];

                    // threshold 未満は完全透明、以上は alpha として使う
                    if (a < threshold * 255.0) a = 0;

                    dst[x * 4 + 0] = b;
                    dst[x * 4 + 1] = g;
                    dst[x * 4 + 2] = r;
                    dst[x * 4 + 3] = a;
                }
            }
        }
        return bgra;
    }

    private Mat PredictMask(Mat bgrFrame)
    {
        using var resized = new Mat();
        Cv2.Resize(bgrFrame, resized, new Size(_inputWidth, _inputHeight), 0, 0, InterpolationFlags.Linear);
        Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

        DenseTensor<float> tensor;
        if (_channelsLast)
        {
            tensor = new DenseTensor<float>(new[] { 1, _inputHeight, _inputWidth, 3 });
            for (int y = 0; y < _inputHeight; y++)
            for (int x = 0; x < _inputWidth; x++)
            {
                var c = resized.At<Vec3b>(y, x);
                tensor[0, y, x, 0] = c.Item0 / 255.0f;
                tensor[0, y, x, 1] = c.Item1 / 255.0f;
                tensor[0, y, x, 2] = c.Item2 / 255.0f;
            }
        }
        else
        {
            tensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });
            for (int y = 0; y < _inputHeight; y++)
            for (int x = 0; x < _inputWidth; x++)
            {
                var c = resized.At<Vec3b>(y, x);
                tensor[0, 0, y, x] = c.Item0 / 255.0f;
                tensor[0, 1, y, x] = c.Item1 / 255.0f;
                tensor[0, 2, y, x] = c.Item2 / 255.0f;
            }
        }

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });
        var output = results.First(r => r.Name == _outputName).AsEnumerable<float>().ToArray();

        // 出力サイズが異なる場合にも対応するため、要素数から正方形に近いサイズを推定
        var pixelCount = output.Length;
        var side = (int)Math.Round(Math.Sqrt(pixelCount));
        int outW = side * side == pixelCount ? side : _inputWidth;
        int outH = side * side == pixelCount ? side : Math.Max(1, pixelCount / outW);

        var mask = new Mat(outH, outW, MatType.CV_8UC1);
        unsafe
        {
            for (int y = 0; y < outH; y++)
            {
                byte* dst = (byte*)mask.Ptr(y).ToPointer();
                for (int x = 0; x < outW; x++)
                {
                    int i = Math.Min(y * outW + x, output.Length - 1);
                    float v = output[i];
                    // logits らしき値なら sigmoid、0..1 ならそのまま
                    if (v < 0 || v > 1) v = 1.0f / (1.0f + MathF.Exp(-v));
                    v = Math.Clamp(v, 0f, 1f);
                    dst[x] = (byte)(v * 255.0f);
                }
            }
        }
        return mask;
    }

    public void Dispose() => _session.Dispose();
}
