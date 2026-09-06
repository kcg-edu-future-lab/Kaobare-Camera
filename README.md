# 顔バレカメラ (Face-Revealing Camera)
[![license](https://img.shields.io/github/license/kcg-edu-future-lab/Kaobare-Camera.svg)](https://github.com/kcg-edu-future-lab/Kaobare-Camera/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/kcg-edu-future-lab/Kaobare-Camera.svg)](https://github.com/kcg-edu-future-lab/Kaobare-Camera/releases)

配信中にこのアプリを起動すると、なんと顔バレすることができて有名になる可能性があります。

## Specification
- PC のカメラから映像を取得し、人物を表示して背景を除去します。

## Usage
- [Releases](https://github.com/kcg-edu-future-lab/Kaobare-Camera/releases) から最新版の `KaobareCamera-x.y.z.zip` をダウンロードして展開し、その中の `KaobareCamera.exe` を実行します。

## Runtime Environments
- Windows 11 or later
  - [.NET 10.0](https://dotnet.microsoft.com/ja-jp/download/dotnet/10.0/runtime) or later

## Release Notes
- **v1.0.6** CPU 使用率を改善。
- **v1.0.5** The first release.

## Third-Party Libraries
- [Reactive Extensions](https://github.com/dotnet/reactive)
- [ReactiveProperty](https://github.com/runceel/ReactiveProperty)
- [ONNX Runtime](https://github.com/Microsoft/onnxruntime)
- [OpenCvSharp](https://github.com/shimat/opencvsharp)

## Third-Party Content
- [MediaPipe Selfie Segmentation](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation)
