#include "CWebsocketServer.hpp"
#include "CNetServer.h"

CWSServer::CWSServer() {
    // Initialize Asio Transport
    m_server.init_asio();
    // Register handler callbacks
    m_server.clear_access_channels(websocketpp::log::alevel::frame_header | websocketpp::log::alevel::frame_payload); 
    m_server.set_open_handler(bind(&CWSServer::on_open,this,::_1));
    m_server.set_close_handler(bind(&CWSServer::on_close,this,::_1));
    m_server.set_message_handler(bind(&CWSServer::on_message,this,::_1,::_2));
    m_server.set_http_handler(bind(&CWSServer::on_http,this,::_1));
    m_port = 9001; //Default port
    m_stop = false;
}
CWSServer::~CWSServer(){
    m_server.stop_listening();
}
void CWSServer::set_port(uint16_t port){
    m_port = port;
}
void CWSServer::excute() {
    // listen on specified port
    m_server.set_reuse_addr(true);
    m_server.listen(m_port);

    // Start the server accept loop
    m_server.start_accept();

    // Start the ASIO io_service run loop
    spdlog::info("websocket 服务开启，正在监听端口 {}",m_port);
    try {
        m_server.run();
    } catch (const std::exception & e) {
        spdlog::error("websocket服务异常");
        std::cout << e.what() << std::endl;
    }
}

void CWSServer::on_open(connection_hdl hdl) {
    {
        lock_guard<mutex> guard(m_action_lock);
        spdlog::info("ws on_open");
        m_actions.push(action(SUBSCRIBE,hdl));
    }
    m_action_cond.notify_one();
}

void CWSServer::on_close(connection_hdl hdl) {
    {
        lock_guard<mutex> guard(m_action_lock);
        spdlog::debug("on_close");
        m_actions.push(action(UNSUBSCRIBE,hdl));
    }
    m_action_cond.notify_one();
}

void CWSServer::on_message(connection_hdl hdl, server::message_ptr msg) {
    // queue message up for sending by processing thread
    {
        spdlog::debug("收到消息:{}",msg->get_payload());
        int64_t tl_time = AMAG::UTILS::get_current_time();
        lock_guard<mutex> guard(m_action_lock);
        spdlog::debug("收到消息，等待消息锁的时间:{}", AMAG::UTILS::get_current_time() - tl_time);
        m_actions.push(action(MESSAGE,hdl,msg));
    }
    m_action_cond.notify_one();
}
void CWSServer::append_msg_extern(std::string msg)
{
    {
        int64_t tl_time = AMAG::UTILS::get_current_time();
        lock_guard<mutex> guard(m_action_lock);
        // spdlog::info("收到UDP消息，等待消息锁的时间:{}", AMAG::UTILS::get_current_time() - tl_time);
        m_actions.push(action(MESSAGE_STR,msg));
    }
    m_action_cond.notify_one();
}
void CWSServer::on_http(websocketpp::connection_hdl hdl) {
    {
        server::connection_ptr con = m_server.get_con_from_hdl(hdl);
        std::string body = con->get_request_body();
        // std::string header_requestId = con->get_request_header("requestId");
        // std::string header_channelId = con->get_request_header("channelId");
        // std::string header_clientCode = con->get_request_header("clientCode");
        // std::string header_requestTime = con->get_request_header("requestTime");
        std::string header = con->get_request_header("header");

        string msg_req = "";
        string msg_rtn="";

        spdlog::info("http body:{}",body);
        // cout << "header:" << header << endl;
        Json::Value body_json = string_to_json(body);
        // Json::Value header_json;

        // header_json["requestId"] = header_requestId;
        // header_json["channelId"] = header_channelId;
        // header_json["clientCode"] = header_clientCode;
        // header_json["requestTime"] = header_requestTime;
        
        // Json::Value msg_req_json;
        
        // if(!header_json || !body_json)
        //     return;


        if(body_json.isMember("command")){ //根据字段command来判断是不是abt调用
            string msg_rtn="";
            ns_ptr->process_http_msg(header, body, msg_rtn);
            con->set_body(msg_rtn);
            con->set_status(websocketpp::http::status_code::ok);
        }
        else{
            // msg_req_json["header"] = body_json["header"];
            // msg_req_json["body"] = body_json["body"];
            // msg_req = json_to_string(msg_req_json);
            ns_ptr->process_tcp_msg(body, msg_rtn);
            // Json::Value msg_rtn_json = string_to_json(msg_rtn);
            // for (auto iter = msg_rtn_json["header"].begin(); iter != msg_rtn_json["header"].end(); iter++)      
            // {          
            //     string key = iter.key().asString();
            //     string value = msg_rtn_json["header"][key].asString();
            //     con->append_header(key,value);
            // }
            con->set_body(msg_rtn);
            con->set_status(websocketpp::http::status_code::ok);
        }
    }
}

action CWSServer::archieve_message(bool &flag) {
    flag = false;
    unique_lock<mutex> lock(m_action_lock);
    while(m_actions.empty()) {
        m_action_cond.wait(lock);
    }
    action a = m_actions.front();
    m_actions.pop();
    lock.unlock();
    if (a.type == SUBSCRIBE) 
    {
        //只有收到SUBSCRIBE的时候，会将handle加入到connection列表中去
        lock_guard<mutex> guard(m_connection_lock);
        m_connections.insert(a.hdl);
    } 
    else if (a.type == UNSUBSCRIBE) 
    {
        //掉线
        lock_guard<mutex> guard(m_connection_lock);
        m_connections.erase(a.hdl);
        if(m_hdl_map.count(a.hdl)>0)
        {
            auto it = m_hdl_map.find(a.hdl);
            if(it != m_hdl_map.end()){
                spdlog::warn("{}下线--unsubscribe",it->second);             
                if(ns_ptr)
                    ns_ptr->off_line(it->second);
                m_hdl_map.erase(it); 
                spdlog::warn("off_line执行完成");
            }
        }
    } 
    else if (a.type == MESSAGE || a.type == MESSAGE_STR) 
    {
        // lock_guard<mutex> guard(m_connection_lock);
        // con_list::iterator it;
        // for (it = m_connections.begin(); it != m_connections.end(); ++it)
        // {
        //     m_server.send(*it,a.msg);
        // }
        // spdlog::info("got_message");
        flag = true;
        a.session_type = "ws";
        return a;
    } 
    else if (a.type == MESSAGE_STR) 
    {
        // lock_guard<mutex> guard(m_connection_lock);
        // con_list::iterator it;
        // for (it = m_connections.begin(); it != m_connections.end(); ++it)
        // {
        //     m_server.send(*it,a.msg);
        // }
        // spdlog::info("got_message");
        flag = true;
        a.session_type = "udp";
        return a;
    } 
    else if (a.type == MESSAGE_HTTP) 
    {
        // lock_guard<mutex> guard(m_connection_lock);
        // con_list::iterator it;
        // for (it = m_connections.begin(); it != m_connections.end(); ++it)
        // {
        //     m_server.send(*it,a.msg);
        // }
        // spdlog::info("got_message");
        flag = true;
        a.session_type = "http";
        return a;
    } 

    else {
        spdlog::warn("undefined websocket message received");// undefined.
    }
    return a;
}
bool CWSServer::send_msg(connection_hdl &send_hdl, const std::string& msg)
{
    unique_lock<mutex> lock(m_send_lock);
    if(m_connections.count(send_hdl)){
        spdlog::debug("send msg: {}",msg);
        m_server.send(send_hdl, msg, websocketpp::frame::opcode::text);
        return true;
    }else{
        spdlog::warn("ws handle已删除, 客户端已经离线, 发送消息{}失败", msg);
        return false;
    }
}
bool CWSServer::bind_hdl_uuid(const connection_hdl &hdl, const std::string &uuid)
{
    lock_guard<mutex> guard(m_connection_lock);
    if(m_hdl_map.count(hdl)==0)
    {
        spdlog::info("bind {} to hdl",uuid);
        m_hdl_map[hdl] = uuid;
    }
    else
    {
        spdlog::warn("uuid {} already combinded to hdl", uuid);
        m_hdl_map[hdl] = uuid;
    }
    return true;
}
void CWSServer::stop_server()
{
    m_stop = true;
    usleep(1000000);
    m_server.stop_listening();
}
void CWSServer::bind_net_server(std::shared_ptr<AMAG::NET::CNetServer> net_ptr)
{
    ns_ptr = net_ptr;
}

