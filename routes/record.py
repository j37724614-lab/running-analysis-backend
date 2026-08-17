import json
import uuid
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any

router = APIRouter()

class RoomMember:
    def __init__(self, member_id: str, ws: WebSocket, camera_index: int = None, is_master: bool = False):
        self.id = member_id
        self.ws = ws
        self.camera_index = camera_index
        self.is_master = is_master
        # 如果不參與錄影 (camera_index 為 None)，預設為就緒
        self.is_ready = camera_index is None

    def to_dict(self):
        return {
            "id": self.id,
            "cameraIndex": self.camera_index,
            "isMaster": self.is_master,
            "isReady": self.is_ready
        }

class Room:
    def __init__(self, room_id: str, expected_camera_count: int):
        self.id = room_id
        self.expected_camera_count = expected_camera_count
        self.members: List[RoomMember] = []
        self.recording_status = "idle"

    async def broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for member in self.members:
            try:
                await member.ws.send_json(message)
            except Exception:
                disconnected.append(member)
        
        for m in disconnected:
            self.members.remove(m)

    async def broadcast_status(self):
        msg = {
            "type": "roomStatus",
            "data": {
                "roomId": self.id,
                "expectedCameraCount": self.expected_camera_count,
                "members": [m.to_dict() for m in self.members]
            }
        }
        await self.broadcast(msg)

class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, Room] = {}

    def create_room(self, expected_camera_count: int) -> str:
        room_id = str(uuid.uuid4())[:6].upper()
        while room_id in self.rooms:
            room_id = str(uuid.uuid4())[:6].upper()
        self.rooms[room_id] = Room(room_id, expected_camera_count)
        return room_id

    def get_room(self, room_id: str) -> Room:
        return self.rooms.get(room_id.upper())

    def remove_empty_rooms(self):
        empty_rooms = [rid for rid, room in self.rooms.items() if not room.members]
        for rid in empty_rooms:
            del self.rooms[rid]

manager = RoomManager()

@router.websocket("/ws")
async def record_websocket(websocket: WebSocket):
    await websocket.accept()
    member_id = str(uuid.uuid4())[:8]
    current_room: Room = None
    current_member: RoomMember = None

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")
            msg_data = msg.get("data", {})

            # print(f"📦 [Test] 收到訊息: {msg_type}", flush=True)
            if msg_type == "sync":
                client_send_time = msg_data.get("clientSendTime", "")
                server_time = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                await websocket.send_json({
                    "type": "sync",
                    "data": {
                        "clientSendTime": client_send_time,
                        "serverTime": server_time
                    }
                })

            elif msg_type == "createRoom":
                expected_camera_count = msg_data.get("expectedCameraCount", 1)
                room_id = manager.create_room(expected_camera_count)
                current_room = manager.get_room(room_id)
                current_member = RoomMember(member_id, websocket, is_master=True)
                current_room.members.append(current_member)
                await current_room.broadcast_status()

            elif msg_type == "joinRoom":
                room_id = msg_data.get("roomId", "").upper()
                camera_index = msg_data.get("cameraIndex")
                is_master = msg_data.get("isMaster", False)

                room = manager.get_room(room_id)
                if room:
                    current_room = room
                    # Check if member already in room
                    existing = next((m for m in room.members if m.id == member_id), None)
                    if existing:
                        existing.camera_index = camera_index
                        existing.is_ready = camera_index is None
                        current_member = existing
                    else:
                        current_member = RoomMember(member_id, websocket, camera_index, is_master)
                        room.members.append(current_member)
                    await room.broadcast_status()
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"找不到房間: {room_id}"}
                    })

            elif msg_type == "startRecording":
                if current_room:
                    scheduled_time = datetime.now(timezone.utc) + timedelta(seconds=2.0)
                    msg_data["scheduledStartTime"] = scheduled_time.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                    
                    current_room.recording_status = "recording"
                    await current_room.broadcast({
                        "type": "startRecording",
                        "data": msg_data
                    })

            elif msg_type == "stopRecording":
                if current_room:
                    current_room.recording_status = "idle"
                    await current_room.broadcast({
                        "type": "stopRecording",
                        "data": {}
                    })

            elif msg_type == "uploadComplete":
                if current_room:
                    # 當其中一個成員完成上傳並取得 runSessionId，廣播給所有人
                    await current_room.broadcast({
                        "type": "uploadComplete",
                        "data": msg_data
                    })
            
            elif msg_type == "updateReady":
                is_ready = msg_data.get("isReady", False)
                if current_member:
                    current_member.is_ready = is_ready
                    if current_room:
                        await current_room.broadcast_status()

            elif msg_type == "cameraPreview":
                if current_room and current_member:
                    # 加入發送者的資訊
                    msg_data["memberId"] = current_member.id
                    msg_data["cameraIndex"] = current_member.camera_index
                    
                    # 僅轉發給 Master
                    for member in current_room.members:
                        if member.is_master:
                            try:
                                await member.ws.send_json({
                                    "type": "cameraPreview",
                                    "data": msg_data
                                })
                            except Exception:
                                pass

            elif msg_type == "requestControl":
                if current_room and current_member:
                    master_member = next((m for m in current_room.members if m.is_master), None)
                    if not master_member:
                        # 房主不存在，直接成為房主
                        current_member.is_master = True
                        await websocket.send_json({
                            "type": "controlGranted",
                            "data": {}
                        })
                        await current_room.broadcast_status()
                    else:
                        # 房主存在，轉發請求給目前房主
                        try:
                            await master_member.ws.send_json({
                                "type": "controlRequest",
                                "data": {
                                    "requesterId": current_member.id,
                                    "cameraIndex": current_member.camera_index
                                }
                            })
                        except Exception:
                            # 房主 websocket 連線異常，直接將控制權給 requester
                            master_member.is_master = False
                            current_member.is_master = True
                            await websocket.send_json({
                                "type": "controlGranted",
                                "data": {}
                            })
                            await current_room.broadcast_status()

            elif msg_type == "respondControlRequest":
                if current_room and current_member and current_member.is_master:
                    requester_id = msg_data.get("requesterId")
                    agree = msg_data.get("agree", False)
                    requester = next((m for m in current_room.members if m.id == requester_id), None)
                    if requester:
                        if agree:
                            # 交換控制權
                            current_member.is_master = False
                            requester.is_master = True
                            
                            # 通知請求者與原房主
                            try:
                                await requester.ws.send_json({
                                    "type": "controlGranted",
                                    "data": {}
                                })
                            except Exception:
                                pass
                            await websocket.send_json({
                                "type": "controlRevoked",
                                "data": {}
                            })
                            await current_room.broadcast_status()
                        else:
                            # 拒絕請求，通知請求者
                            try:
                                await requester.ws.send_json({
                                    "type": "controlRejected",
                                    "data": {"message": "房主拒絕了您的主控權要求"}
                                })
                            except Exception:
                                pass

    except WebSocketDisconnect:
        if current_room and current_member in current_room.members:
            current_room.members.remove(current_member)
            if not current_room.members:
                if current_room.id in manager.rooms:
                    del manager.rooms[current_room.id]
            else:
                await current_room.broadcast_status()
    except Exception as e:
        print(f"WebSocket error: {e}", flush=True)
        if current_room and current_member in current_room.members:
            current_room.members.remove(current_member)
            await current_room.broadcast_status()
