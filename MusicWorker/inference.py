import wget
import os
import infer_web
import scipy.io.wavfile as wavfile


class Inference:
    inference_count = 0

    def __init__(
        self,
        model_name=None,
        source_audio_path=None,
        output_file_name=None,
        feature_index_path="",
        f0_file=None,
        speaker_id=0,
        transposition=0,
        f0_method="harvest",
        crepe_hop_length=160,
        harvest_median_filter=3,
        resample=0,
        mix=1,
        feature_ratio=0.78,
        protection_amnt=0.33,
        protect1=False,
    ):
        Inference.inference_count += 1
        self._model_name = model_name
        self._source_audio_path = source_audio_path
        self._output_file_name = output_file_name
        self._feature_index_path = feature_index_path
        self._f0_file = f0_file
        self._speaker_id = speaker_id
        self._transposition = transposition
        self._f0_method = f0_method
        self._crepe_hop_length = crepe_hop_length
        self._harvest_median_filter = harvest_median_filter
        self._resample = resample
        self._mix = mix
        self._feature_ratio = feature_ratio
        self._protection_amnt = protection_amnt
        self._protect1 = protect1
        self._id = Inference.inference_count

        if not os.path.exists("./hubert_base.pt"):
            print("Downloading Hubert")
            wget.download(
                "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
                out="./hubert_base.pt",
            )

        if not os.path.exists("./rmvpe.pt"):
            wget.download(
                "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
                out="./rmvpe.pt",
            )

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def audio(self):
        return self._audio

    @audio.setter
    def audio_file(self, audio):
        self._audio_file = audio

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name

    @property
    def source_audio_path(self):
        return self._source_audio_path

    @source_audio_path.setter
    def source_audio_path(self, source_audio_path):
        if not self._output_file_name:
            self._output_file_name = os.path.join(
                "./audio-outputs", os.path.basename(source_audio_path)
            )
        self._source_audio_path = source_audio_path

    @property
    def output_file_name(self):
        return self._output_file_name

    @output_file_name.setter
    def output_file_name(self, output_file_name):
        self._output_file_name = output_file_name

    @property
    def feature_index_path(self):
        return self._feature_index_path

    @feature_index_path.setter
    def feature_index_path(self, feature_index_path):
        self._feature_index_path = feature_index_path

    @property
    def f0_file(self):
        return self._f0_file

    @f0_file.setter
    def f0_file(self, f0_file):
        self._f0_file = f0_file

    @property
    def speaker_id(self):
        return self._speaker_id

    @speaker_id.setter
    def speaker_id(self, speaker_id):
        self._speaker_id = speaker_id

    @property
    def transposition(self):
        return self._transposition

    @transposition.setter
    def transposition(self, transposition):
        self._transposition = transposition

    @property
    def f0_method(self):
        return self._f0_method

    @f0_method.setter
    def f0_method(self, f0_method):
        self._f0_method = f0_method

    @property
    def crepe_hop_length(self):
        return self._crepe_hop_length

    @crepe_hop_length.setter
    def crepe_hop_length(self, crepe_hop_length):
        self._crepe_hop_length = crepe_hop_length

    @property
    def harvest_median_filter(self):
        return self._harvest_median_filter

    @crepe_hop_length.setter
    def harvest_median_filter(self, harvest_median_filter):
        self._harvest_median_filter = harvest_median_filter

    @property
    def resample(self):
        return self._resample

    @resample.setter
    def resample(self, resample):
        self._resample = resample

    @property
    def mix(self):
        return self._mix

    @mix.setter
    def mix(self, mix):
        self._mix = mix

    @property
    def feature_ratio(self):
        return self._feature_ratio

    @feature_ratio.setter
    def feature_ratio(self, feature_ratio):
        self._feature_ratio = feature_ratio

    @property
    def protection_amnt(self):
        return self._protection_amnt

    @protection_amnt.setter
    def protection_amnt(self, protection_amnt):
        self._protection_amnt = protection_amnt

    @property
    def protect1(self):
        return self._protect1

    @protect1.setter
    def protect1(self, protect1):
        self._protect1 = protect1

    def run(self):
        index = "./weights/added_IVF617_Flat_nprobe_1_SiriVT_v1.index"
        pth = "./weights/SiriVT.pth"

        infer_web.get_vc(pth)

        conversion_data = infer_web.vc_single(
            self.speaker_id,
            self.source_audio_path,
            self.source_audio_path,
            self.transposition,
            self.f0_file,
            self.f0_method,
            index,
            index,
            self.feature_ratio,
            self.harvest_median_filter,
            self.resample,
            self.mix,
            self.protection_amnt,
            self.crepe_hop_length,
        )

        if "Success." in conversion_data[0]:
            print(f"Writing to {self._output_file_name}")
            wavfile.write(
                "%s/%s" % ("audio-outputs", os.path.basename(self._output_file_name)),
                conversion_data[1][0],
                conversion_data[1][1],
            )
            return {"success": True, "file": self._output_file_name}
        else:
            return {"success": False, "file": self._output_file_name}
