# Third-party notices

The Reference branch integrates three separately obtained runtimes:

- showlab/RobotSeg for whole-robot segmentation. Its repository declares the
  Apache License 2.0.
- advimman/LaMa and a Big-LaMa checkpoint for image inpainting. Its repository
  declares the Apache License 2.0.
- Ultralytics YOLOE for residual-robot quality assurance after inpainting.
  Ultralytics is distributed under the GNU Affero General Public License v3.0
  (AGPL-3.0), with commercial licensing available separately.

These source trees and all model weights are not vendored in this repository.
Users are responsible for obtaining them and complying with the terms attached
to each source distribution and checkpoint. The Apache-2.0 license in this
repository applies only to this repository's own code and does not relicense
third-party software or model weights.

Project links:

- RobotSeg: https://github.com/showlab/RobotSeg
- LaMa: https://github.com/advimman/lama
- YOLOE: https://github.com/THU-MIG/yoloe
- Ultralytics: https://github.com/ultralytics/ultralytics
