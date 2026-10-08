class Hivebear < Formula
  desc "Run local AI models, with picks matched to your hardware"
  homepage "https://hivebear.com"
  version "0.1.9"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-apple-darwin.tar.gz"
      sha256 "f2e71451d2ce1b7b614cdf7e1cad316510245c8660772107b814e387ce0e0672"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-apple-darwin.tar.gz"
      sha256 "8a7c81cd194d0df3d5f3873f15f7ee4e5ac86006c54434c3b15417affdf6efbd"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "13069fba9a56d412ea4a2a970cfe5746bc1b4d1fc70bc5d9e8d5370ef5657eb7"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "0a13a8cd0f89bd5d182ce47ab89f5259647976fe248fcf654518bfe2e54d49af"
    end
  end

  def install
    bin.install "hivebear"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/hivebear --version")
  end

  def caveats
    <<~EOS
      Get started with HiveBear:

        hivebear quickstart

      This will profile your hardware, recommend the best model,
      download it, and start an interactive chat session.
    EOS
  end
end
