// Copyright (C) 2024  MAlba124 <marlhan@proton.me>
//
// StreamCraft is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// StreamCraft is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with StreamCraft.  If not, see <https://www.gnu.org/licenses/>.

use crate::{
    debug, element_def,
    element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sink, Srcs},
    error, info,
    pipeline::{error::Error, Data, Datagram, Message, Parent, SinkPipe},
};

use crossbeam_channel::{bounded, unbounded, Receiver};
use libav::demuxing::{CodecID, CodecParams, Demuxer, ResourceLocation};

///```text
/// +------------------+
/// |             _____|
/// |            | src |----> AVPacket
/// |  DemuxSrc   -----|
/// |            | src |----> AVPacket
/// |             ^^^^^|
/// +------------------+
///```
pub struct DemuxSrc {
    demuxer: Demuxer,
    video_sink: SinkPipe,
    video_stream_index: i32,
    audio_sink: SinkPipe,
    audio_stream_index: i32,
    parent: Parent,
}

impl DemuxSrc {
    pub fn new(resource: ResourceLocation) -> Result<Self, Error> {
        let demuxer = Demuxer::new(resource).map_err(|e| Error::AVError(e))?;

        Ok(Self {
            demuxer,
            audio_sink: SinkPipe::default(),
            video_stream_index: -1,
            video_sink: SinkPipe::default(),
            audio_stream_index: -1,
            parent: Parent::default(),
        })
    }

    pub fn link_video_sink_element(
        &mut self,
        stream_index: i32,
        sink: impl Element + 'static,
    ) -> Result<(), Error> {
        if sink.get_sink_type() != ElementType::AVPacketVideoSink {
            return Err(Error::InvalidSinkType);
        }

        if let Sink::One(format) = sink.get_architecture().sink {
            if format == CommonFormat::AVPacket {
                self.video_sink.set_element(sink);
                self.video_stream_index = stream_index;
                return Ok(());
            }
        }

        Err(Error::InvalidSinkType)
    }

    pub fn link_audio_sink_element(
        &mut self,
        stream_index: i32,
        sink: impl Element + 'static,
    ) -> Result<(), Error> {
        if sink.get_sink_type() != ElementType::AVPacketAudioSink {
            return Err(Error::InvalidSinkType);
        }

        if let Sink::One(format) = sink.get_architecture().sink {
            if format == CommonFormat::AVPacket {
                self.audio_sink.set_element(sink);
                self.audio_stream_index = stream_index;
                return Ok(());
            }
        }

        Err(Error::InvalidSinkType)
    }

    pub fn get_video_stream(&self) -> Result<(i32, CodecID, CodecParams), Error> {
        self.demuxer
            .get_video_stream()
            .map_err(|e| Error::AVError(e))
    }

    pub fn get_audio_stream(&self) -> Result<(i32, CodecID, CodecParams), Error> {
        self.demuxer
            .get_audio_stream()
            .map_err(|e| Error::AVError(e))
    }

    fn run_loop(&mut self) -> bool {
        match self.demuxer.read_frame() {
            Ok(packet) => {
                let stream_index = packet.stream_index();
                if stream_index == self.audio_stream_index {
                    info!("Got audio packet");
                    if let Err(e) = self
                        .audio_sink
                        .send_datagram(Datagram::Data(Data::AVPacket(packet)))
                    {
                        error!("{e}");
                        return false;
                    }
                } else if stream_index == self.video_stream_index {
                    info!("Got video packet");
                    if let Err(e) = self
                        .video_sink
                        .send_datagram(Datagram::Data(Data::AVPacket(packet)))
                    {
                        error!("{e}");
                        return false;
                    }
                }
            }
            Err(e) => {
                error!("{e}");
                return false;
            }
        }

        true
    }

    fn init(&mut self) -> Result<(), Error> {
        // Video
        {
            let (datagram_sender, datagram_receiver) = bounded(0);
            let (msg_sender, my_msg_receiver) = unbounded();
            let parent = Parent::new(msg_sender);
            let mut sink_element = self.video_sink.take_element()?;
            sink_element.set_parent(parent);
            let datagram_receiver_clone = datagram_receiver.clone();

            self.video_sink.thread_handle = Some(std::thread::spawn(move || {
                match sink_element.run(datagram_receiver_clone) {
                    Ok(_) => {}
                    Err(e) => error!("Error occurred running sink element: {e}"),
                }
            }));
            self.video_sink.msg_receiver = Some(my_msg_receiver);
            self.video_sink.datagram_sender = Some(datagram_sender);
        }

        // Audio
        {
            let (datagram_sender, datagram_receiver) = bounded(0);
            let (msg_sender, my_msg_receiver) = unbounded();
            let parent = Parent::new(msg_sender);
            let mut sink_element = self.audio_sink.take_element()?;
            sink_element.set_parent(parent);
            let datagram_receiver_clone = datagram_receiver.clone();

            self.audio_sink.thread_handle = Some(std::thread::spawn(move || {
                match sink_element.run(datagram_receiver_clone) {
                    Ok(_) => {}
                    Err(e) => error!("Error occurred running sink element: {e}"),
                }
            }));
            self.audio_sink.msg_receiver = Some(my_msg_receiver);
            self.audio_sink.datagram_sender = Some(datagram_sender);
        }

        Ok(())
    }
}

impl Element for DemuxSrc {
    fn get_sink_type(&self) -> ElementType {
        ElementType::AVPacketSrc
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::None,
            srcs: Srcs::Two((CommonFormat::AVPacket, CommonFormat::AVPacket)),
        }
    }

    fn run(&mut self, parent_datagram_receiver: Receiver<Datagram>) -> Result<(), Error> {
        self.init()?;

        loop {
            match parent_datagram_receiver
                .recv()
                .map_err(|_| Error::FailedToRecvFromParent)?
            {
                Datagram::Message(msg) => match msg {
                    Message::Iter => {
                        if !self.run_loop() {
                            break;
                        }
                        self.parent.send_iter_fin()?;
                    }
                    Message::Quit => break,
                    _ => return Err(Error::ReceivedInvalidDatagramFromParent),
                },
                _ => return Err(Error::ReceivedInvalidDatagramFromParent),
            }

            while let Some(_msg) = self.video_sink.try_recv_msg()? {
                // TODO: Handle messages
            }
            while let Some(_msg) = self.audio_sink.try_recv_msg()? {
                // TODO: Handle messages
            }
        }

        self.parent.send_finished()
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }

    fn cleanup(&mut self) -> Result<(), Error> {
        if self.video_sink.is_operational() {
            if let Err(e) = self.video_sink.send_quit() {
                error!("Failed to send quit to video sink: {e}");
            }
            self.video_sink.drop_data_sender();
        }
        if self.audio_sink.is_operational() {
            if let Err(e) = self.audio_sink.send_quit() {
                error!("Failed to send quit to audio sink: {e}");
            }
            self.audio_sink.drop_data_sender();
        }

        if self.video_sink.is_operational() {
            self.video_sink.join_thread()?;
        }
        if self.audio_sink.is_operational() {
            self.audio_sink.join_thread()?;
        }

        Ok(())
    }
}

element_def! {
    DemuxSrc,
    "demuxsrc"
}

#[cfg(test)]
mod tests {
    use super::*;
}
