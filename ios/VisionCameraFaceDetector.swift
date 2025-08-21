import Vision
import UIKit
import AVFoundation
import Vision
import CoreML
import CoreGraphics
import CoreImage

@objc(VisionCameraFaceDetector)
public class VisionCameraFaceDetector: FrameProcessorPlugin {

  @objc
  public override init(proxy: VisionCameraProxyHolder, options: [AnyHashable : Any]! = [:]) {
    super.init(proxy: proxy, options: options)
  }

  // MARK: - Helpers

  /// convert radians to degrees
  private func deg(_ rad: Double?) -> Double? {
    guard let r = rad else { return nil }
    return r * 180.0 / .pi
  }

  /// Vision normalized rect → UIKit coordinates (origin top-left)
  private func denormalizeRect(_ r: CGRect, imgW: Int, imgH: Int) -> CGRect {
    let x = r.origin.x * CGFloat(imgW)
    let y = (1.0 - r.origin.y - r.size.height) * CGFloat(imgH) // flip Y
    let w = r.size.width * CGFloat(imgW)
    let h = r.size.height * CGFloat(imgH)
    return CGRect(x: x, y: y, width: w, height: h)
  }

  /// Convert landmark region to array of points
  private func points(from region: VNFaceLandmarkRegion2D?,
                      faceBBox: CGRect,
                      imgW: Int, imgH: Int) -> [[String: CGFloat]] {
    guard let region = region else { return [] }
    return region.normalizedPoints.map { p in
      // normalized relative to face box
      let nx = faceBBox.origin.x + CGFloat(p.x) * faceBBox.size.width
      let ny = faceBBox.origin.y + CGFloat(p.y) * faceBBox.size.height
      let x = nx * CGFloat(imgW)
      let y = (1.0 - ny) * CGFloat(imgH)
      return ["x": x, "y": y]
    }
  }

  private func boundingBoxDict(from pixelRect: CGRect) -> [String: Any] {
    return [
      "x": pixelRect.origin.x,
      "y": pixelRect.origin.y,
      "width": pixelRect.width,
      "height": pixelRect.height,
      "boundingCenterX": pixelRect.midX,
      "boundingCenterY": pixelRect.midY
    ]
  }

  private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
      let dx = p1.x - p2.x
      let dy = p1.y - p2.y
      return sqrt(dx*dx + dy*dy)
  }

  private func estimateEyeOpen(eye: VNFaceLandmarkRegion2D?) -> CGFloat? {
      guard let eye = eye, eye.pointCount > 4 else { return nil }
      let points = (0..<eye.pointCount).map { eye.normalizedPoints[$0] }

      // Góc trái & phải của mắt
      let leftCorner = points.first!
      let rightCorner = points[points.count/2]

      // Điểm trên & dưới (lấy trung bình vài point)
      let upper = points[1]
      let lower = points[points.count - 2]

      let eyeWidth = distance(leftCorner, rightCorner)
      let eyeHeight = distance(upper, lower)

      guard eyeWidth > 0 else { return nil }
      let ratio = eyeHeight / eyeWidth

      // Chuẩn hóa: mắt mở ~0.3–0.35 là bình thường
      return min(max((ratio - 0.15) / 0.2, 0), 1)
  }

  private func estimateSmile(lips: VNFaceLandmarkRegion2D?) -> CGFloat? {
      guard let lips = lips, lips.pointCount > 5 else { return nil }
      let points = (0..<lips.pointCount).map { lips.normalizedPoints[$0] }

      let leftCorner = points.first!
      let rightCorner = points[points.count/2]
      let topMid = points[points.count/4]
      let bottomMid = points[3*points.count/4]

      let mouthWidth = distance(leftCorner, rightCorner)
      let mouthOpen = distance(topMid, bottomMid)

      guard mouthWidth > 0 else { return nil }
      let ratio = mouthOpen / mouthWidth

      // Chuẩn hóa: môi cong lên nhiều thì ratio lớn hơn
      return min(max((ratio - 0.2) / 0.3, 0), 1)
  }

  private func estimatePitch(face: VNFaceObservation) -> Float {
      guard let landmarks = face.landmarks,
            let nose = landmarks.nose?.normalizedPoints,
            let leftEye = landmarks.leftEye?.normalizedPoints,
            let rightEye = landmarks.rightEye?.normalizedPoints,
            let mouth = landmarks.outerLips?.normalizedPoints else {
          return 0.0
      }

      // Lấy điểm trung tâm mắt trái và mắt phải
      let leftEyeCenter = averagePoint(points: leftEye)
      let rightEyeCenter = averagePoint(points: rightEye)
      let eyeCenterY = (leftEyeCenter.y + rightEyeCenter.y) / 2.0

      // Lấy điểm trung tâm miệng
      let mouthCenter = averagePoint(points: mouth)

      // Khoảng cách dọc giữa mắt và miệng
      let eyeMouthDist = abs(mouthCenter.y - eyeCenterY)

      // Ước lượng chiều cao khuôn mặt
      let faceHeight = face.boundingBox.height

      // Tính ratio (chuẩn hóa về khoảng -30° ~ +30°)
      let ratio = Float(eyeMouthDist / faceHeight)

      // Convert sang độ pitch (gật đầu)
      // Giá trị này chỉ tương đối, bạn có thể scale lại để hợp lý hơn
      let pitch = (0.5 - ratio) * 60.0

      return pitch
  }

  private func averagePoint(points: [CGPoint]) -> CGPoint {
      guard !points.isEmpty else { return .zero }
      let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
      return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
  }

  private func estimateBrightness(from sampleBuffer: CMSampleBuffer) -> Float {
      guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
          return 0.0
      }
      let ciImage = CIImage(cvPixelBuffer: imageBuffer)

      // Dùng CIFilter để tính độ sáng trung bình
      let extent = ciImage.extent
      let filter = CIFilter(name: "CIAreaAverage",
                            parameters: [kCIInputImageKey: ciImage,
                                         kCIInputExtentKey: CIVector(cgRect: extent)])!

      guard let outputImage = filter.outputImage else {
          return 0.0
      }

      var bitmap = [UInt8](repeating: 0, count: 4)
      let context = CIContext(options: [.workingColorSpace: NSNull()])
      context.render(outputImage,
                     toBitmap: &bitmap,
                     rowBytes: 4,
                     bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                     format: .RGBA8,
                     colorSpace: nil)

      // Chuyển RGB sang giá trị sáng [0..1]
      let r = Float(bitmap[0]) / 255.0
      let g = Float(bitmap[1]) / 255.0
      let b = Float(bitmap[2]) / 255.0
      let brightness = (r + g + b) / 3.0

      return brightness
  }

  // MARK: - Callback

  @objc
  public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable : Any]?) -> Any? {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(frame.buffer) else { return nil }

    let imgW = CVPixelBufferGetWidth(pixelBuffer)
    let imgH = CVPixelBufferGetHeight(pixelBuffer)

    var faceAttributes: [Any] = []

    let request = VNDetectFaceLandmarksRequest { req, err in
      guard err == nil, let faces = req.results as? [VNFaceObservation], !faces.isEmpty else { return }

      for face in faces {
        let bbox = self.denormalizeRect(face.boundingBox, imgW: imgW, imgH: imgH)
        let landmark = face.landmarks
        let brightness = self.estimateBrightness(from: frame.buffer)
        var contours: [String: [[String: CGFloat]]] = [:]
        contours["FACE"] = self.points(from: landmark?.faceContour, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["LEFT_EYEBROW_TOP"] = self.points(from: landmark?.leftEyebrow, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["LEFT_EYEBROW_BOTTOM"] = contours["LEFT_EYEBROW_TOP"] ?? []
        contours["RIGHT_EYEBROW_TOP"] = self.points(from: landmark?.rightEyebrow, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["RIGHT_EYEBROW_BOTTOM"] = contours["RIGHT_EYEBROW_TOP"] ?? []
        contours["LEFT_EYE"] = self.points(from: landmark?.leftEye, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["RIGHT_EYE"] = self.points(from: landmark?.rightEye, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["NOSE_BRIDGE"] = self.points(from: landmark?.noseCrest, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["NOSE_BOTTOM"] = self.points(from: landmark?.nose, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["UPPER_LIP_TOP"] = self.points(from: landmark?.outerLips, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["UPPER_LIP_BOTTOM"] = self.points(from: landmark?.innerLips, faceBBox: face.boundingBox, imgW: imgW, imgH: imgH)
        contours["LOWER_LIP_TOP"] = contours["UPPER_LIP_BOTTOM"] ?? []
        contours["LOWER_LIP_BOTTOM"] = contours["UPPER_LIP_TOP"] ?? []
        if let leftEyePoints = landmark?.leftEye?.normalizedPoints,
           let faceContourPoints = landmark?.faceContour?.normalizedPoints {

            if let leftEyeOuter = leftEyePoints.first,   // điểm ngoài cùng bên trái mắt
               let jawLeft = faceContourPoints.first {   // điểm đầu tiên trong contour (hàm trái)

                let cheekLeftX = (leftEyeOuter.x + jawLeft.x) / 2
                let cheekLeftY = (leftEyeOuter.y + jawLeft.y) / 2
                contours["LEFT_CHEEK"] = [
                    ["x": cheekLeftX * face.boundingBox.width * CGFloat(imgW) + face.boundingBox.minX * CGFloat(imgW),
                     "y": (1 - (cheekLeftY * face.boundingBox.height + face.boundingBox.minY)) * CGFloat(imgH)]
                ]
            }
        }

        if let rightEyePoints = landmark?.rightEye?.normalizedPoints,
           let faceContourPoints = landmark?.faceContour?.normalizedPoints {

            if let rightEyeOuter = rightEyePoints.last,   // điểm ngoài cùng bên phải mắt
               let jawRight = faceContourPoints.last {    // điểm cuối trong contour (hàm phải)

                let cheekRightX = (rightEyeOuter.x + jawRight.x) / 2
                let cheekRightY = (rightEyeOuter.y + jawRight.y) / 2
                contours["RIGHT_CHEEK"] = [
                    ["x": cheekRightX * face.boundingBox.width * CGFloat(imgW) + face.boundingBox.minX * CGFloat(imgW),
                     "y": (1 - (cheekRightY * face.boundingBox.height + face.boundingBox.minY)) * CGFloat(imgH)]
                ]
            }
        }

        var map: [String: Any] = [:]
        map["rollAngle"] = self.deg(face.roll?.doubleValue) ?? NSNull()
        map["pitchAngle"] = self.estimatePitch(face: face)
        map["yawAngle"] = self.deg(face.yaw?.doubleValue) ?? NSNull()

        if let leftProb = self.estimateEyeOpen(eye: landmark?.leftEye) {
            map["leftEyeOpenProbability"] = leftProb
        } else {
            map["leftEyeOpenProbability"] = NSNull()
        }

        if let rightProb = self.estimateEyeOpen(eye: landmark?.rightEye) {
            map["rightEyeOpenProbability"] = rightProb
        } else {
            map["rightEyeOpenProbability"] = NSNull()
        }

        if let smileProb = self.estimateSmile(lips: landmark?.outerLips) {
            map["smilingProbability"] = smileProb
        } else {
            map["smilingProbability"] = NSNull()
        }
        map["bounds"] = self.boundingBoxDict(from: bbox)
        map["contours"] = contours
        map["brightness"] = brightness * 100
        faceAttributes.append(map)
      }
    }

    let handler = VNImageRequestHandler(cmSampleBuffer: frame.buffer, orientation: .up, options: [:])
    do {
      try handler.perform([request])
    } catch {
      return nil
    }

    return faceAttributes
  }
}
