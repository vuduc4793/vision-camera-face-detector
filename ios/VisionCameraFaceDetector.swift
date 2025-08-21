import Vision
import UIKit
import AVFoundation
import CoreML
import CoreGraphics
import CoreImage

@objc(VisionCameraFaceDetector)
public class VisionCameraFaceDetector: FrameProcessorPlugin {

  // MARK: - Init

  @objc
  public override init(proxy: VisionCameraProxyHolder, options: [AnyHashable: Any]! = [:]) {
    super.init(proxy: proxy, options: options)
  }

  // MARK: - Utilities

  private func degrees(from radians: Double?) -> Double? {
    guard let radians else { return nil }
    return radians * 180.0 / .pi
  }

  private func denormalizeRect(_ rect: CGRect, imgW: Int, imgH: Int) -> CGRect {
    CGRect(
      x: rect.origin.x * CGFloat(imgW),
      y: (1.0 - rect.origin.y - rect.height) * CGFloat(imgH),
      width: rect.width * CGFloat(imgW),
      height: rect.height * CGFloat(imgH)
    )
  }

  private func landmarkPoints(
    from region: VNFaceLandmarkRegion2D?,
    faceBBox: CGRect,
    imgW: Int, imgH: Int
  ) -> [[String: CGFloat]] {
    guard let region else { return [] }

    return region.normalizedPoints.map { p in
      let nx = faceBBox.origin.x + CGFloat(p.x) * faceBBox.width
      let ny = faceBBox.origin.y + CGFloat(p.y) * faceBBox.height
      return [
        "x": nx * CGFloat(imgW),
        "y": (1.0 - ny) * CGFloat(imgH)
      ]
    }
  }

  private func boundingBoxDict(_ rect: CGRect) -> [String: Any] {
    [
      "x": rect.origin.x,
      "y": rect.origin.y,
      "width": rect.width,
      "height": rect.height,
      "boundingCenterX": rect.midX,
      "boundingCenterY": rect.midY
    ]
  }

  private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
    hypot(p1.x - p2.x, p1.y - p2.y)
  }

  private func averagePoint(_ points: [CGPoint]) -> CGPoint {
    guard !points.isEmpty else { return .zero }
    let sum = points.reduce(.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
    return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
  }

  // MARK: - Estimators

  /// Ước lượng mức mở mắt [0..1], trả nil nếu landmark không đủ điểm
  private func estimateEyeOpen(eye: VNFaceLandmarkRegion2D?) -> CGFloat? {
    guard let eye, eye.pointCount > 4 else { return nil }

    let points = eye.normalizedPoints
    let leftCorner = points.first!
    let rightCorner = points[points.count / 2]
    let upper = points[1]
    let lower = points[points.count - 2]

    let width = distance(leftCorner, rightCorner)
    let height = distance(upper, lower)

    guard width > 0 else { return nil }
    let ratio = height / width

    // Chuẩn hóa
    let normalized = min(max((ratio - 0.15) / 0.2, 0), 1)

    return normalized
  }

  /// Trả về trạng thái OPEN / CLOSED / UNKNOWN dựa trên prob
  private func eyeState(from prob: CGFloat?) -> String {
    guard let prob else { return "UNKNOWN" }
    return prob < 0.2 ? "CLOSED" : "OPEN"
  }

  private func estimateSmile(lips: VNFaceLandmarkRegion2D?) -> CGFloat? {
    guard let lips, lips.pointCount > 5 else { return nil }

    let points = lips.normalizedPoints
    let leftCorner = points.first!
    let rightCorner = points[points.count / 2]
    let topMid = points[points.count / 4]
    let bottomMid = points[3 * points.count / 4]

    let width = distance(leftCorner, rightCorner)
    let height = distance(topMid, bottomMid)

    guard width > 0 else { return nil }
    let ratio = height / width

    return min(max((ratio - 0.2) / 0.3, 0), 1)
  }

  private func estimatePitch(face: VNFaceObservation) -> Float {
    guard let landmarks = face.landmarks,
          let leftEye = landmarks.leftEye?.normalizedPoints,
          let rightEye = landmarks.rightEye?.normalizedPoints,
          let mouth = landmarks.outerLips?.normalizedPoints else {
      return 0.0
    }

    let eyeCenterY = (averagePoint(leftEye).y + averagePoint(rightEye).y) / 2.0
    let mouthCenterY = averagePoint(mouth).y
    let eyeMouthDist = abs(mouthCenterY - eyeCenterY)
    let ratio = Float(eyeMouthDist / face.boundingBox.height)

    return (0.5 - ratio) * 60.0
  }

  private func estimateBrightness(from sampleBuffer: CMSampleBuffer) -> Float {
    guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return 0 }

    let ciImage = CIImage(cvPixelBuffer: imageBuffer)
    let filter = CIFilter(
      name: "CIAreaAverage",
      parameters: [
        kCIInputImageKey: ciImage,
        kCIInputExtentKey: CIVector(cgRect: ciImage.extent)
      ]
    )

    guard let outputImage = filter?.outputImage else { return 0 }

    var bitmap = [UInt8](repeating: 0, count: 4)
    CIContext(options: [.workingColorSpace: NSNull()]).render(
      outputImage,
      toBitmap: &bitmap,
      rowBytes: 4,
      bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
      format: .RGBA8,
      colorSpace: nil
    )

    let (r, g, b) = (Float(bitmap[0]), Float(bitmap[1]), Float(bitmap[2]))
    return (r + g + b) / (3 * 255.0)
  }

  // MARK: - Callback

  @objc
  public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable: Any]?) -> Any? {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(frame.buffer) else { return nil }

    let imgW = CVPixelBufferGetWidth(pixelBuffer)
    let imgH = CVPixelBufferGetHeight(pixelBuffer)
    var faceAttributes: [[String: Any]] = []

    let request = VNDetectFaceLandmarksRequest { req, err in
      guard err == nil, let faces = req.results as? [VNFaceObservation], !faces.isEmpty else { return }

      for face in faces {
        let bbox = self.denormalizeRect(face.boundingBox, imgW: imgW, imgH: imgH)
        let landmarks = face.landmarks
        var contours: [String: [[String: CGFloat]]] = [:]

        func addContour(_ key: String, _ region: VNFaceLandmarkRegion2D?) {
          contours[key] = self.landmarkPoints(from: region, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        }

        addContour("FACE", landmarks?.faceContour)
        addContour("LEFT_EYEBROW_TOP", landmarks?.leftEyebrow)
        contours["LEFT_EYEBROW_BOTTOM"] = contours["LEFT_EYEBROW_TOP"]
        addContour("RIGHT_EYEBROW_TOP", landmarks?.rightEyebrow)
        contours["RIGHT_EYEBROW_BOTTOM"] = contours["RIGHT_EYEBROW_TOP"]
        addContour("LEFT_EYE", landmarks?.leftEye)
        addContour("RIGHT_EYE", landmarks?.rightEye)
        addContour("NOSE_BRIDGE", landmarks?.noseCrest)
        addContour("NOSE_BOTTOM", landmarks?.nose)
        addContour("UPPER_LIP_TOP", landmarks?.outerLips)
        addContour("UPPER_LIP_BOTTOM", landmarks?.innerLips)
        contours["LOWER_LIP_TOP"] = contours["UPPER_LIP_BOTTOM"]
        contours["LOWER_LIP_BOTTOM"] = contours["UPPER_LIP_TOP"]

        let brightness = self.estimateBrightness(from: frame.buffer)

        // Eye probability & state
        let leftProb = self.estimateEyeOpen(eye: landmarks?.leftEye)
        let rightProb = self.estimateEyeOpen(eye: landmarks?.rightEye)

        var map: [String: Any] = [
          "rollAngle": self.degrees(from: face.roll?.doubleValue) ?? NSNull(),
          "pitchAngle": self.estimatePitch(face: face),
          "yawAngle": self.degrees(from: face.yaw?.doubleValue) ?? NSNull(),
          "bounds": self.boundingBoxDict(bbox),
          "contours": contours,
          "brightness": brightness * 100,
          "leftEyeOpenProbability": leftProb ?? NSNull(),
          "rightEyeOpenProbability": rightProb ?? NSNull(),
          "leftEyeState": self.eyeState(from: leftProb),
          "rightEyeState": self.eyeState(from: rightProb),
          "smilingProbability": self.estimateSmile(lips: landmarks?.outerLips) ?? NSNull()
        ]

        faceAttributes.append(map)
      }
    }

    let handler = VNImageRequestHandler(cmSampleBuffer: frame.buffer, orientation: .up)
    do { try handler.perform([request]) } catch { return nil }

    return faceAttributes
  }
}
