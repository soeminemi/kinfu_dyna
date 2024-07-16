#include "CWebsocketServer.hpp"

CWSServer::CWSServer() {
    // Initialize Asio Transport
    m_server.init_asio();
    // Register handler callbacks
    m_server.clear_access_channels(websocketpp::log::alevel::frame_header | websocketpp::log::alevel::frame_payload); 
    m_server.set_open_handler(bind(&CWSServer::on_open,this,::_1));
    m_server.set_close_handler(bind(&CWSServer::on_close,this,::_1));
    m_server.set_message_handler(bind(&CWSServer::on_message,this,::_1,::_2));
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
    std::cout<<"websocket 服务开启，正在监听端口" <<m_port<<std::endl;
    try {
        m_server.run();
    } catch (const std::exception & e) {
        std::cout<<"websocket服务异常"<<std::endl;
        std::cout << e.what() << std::endl;
    }
}

void CWSServer::on_open(connection_hdl hdl) {
    {
        lock_guard<mutex> guard(m_action_lock);
        std::cout<<"ws on_open"<<std::endl;
        m_actions.push(action(SUBSCRIBE,hdl));
    }
    m_action_cond.notify_one();
}

void CWSServer::on_close(connection_hdl hdl) {
    {
        lock_guard<mutex> guard(m_action_lock);
        std::cout<<"ws server on_close"<<std::endl;
        m_actions.push(action(UNSUBSCRIBE,hdl));
    }
    m_action_cond.notify_one();
}

void CWSServer::on_message(connection_hdl hdl, server::message_ptr msg) {
    // queue message up for sending by processing thread
    {
        // std::cout<<"收到消息:"<<msg->get_payload()<<std::endl;
        lock_guard<mutex> guard(m_action_lock);
        m_actions.push(action(MESSAGE,hdl,msg));
    }
    m_action_cond.notify_one();
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
                std::cout<<it->second<<" 下线--unsubscribe"<<std::endl;
                m_hdl_map.erase(it); 
                std::cout<<"off_line执行完成"<<std::endl;
            }
        }
    } 
    else if (a.type == MESSAGE || a.type == MESSAGE_STR) 
    {
        flag = true;
        a.session_type = "ws";
        return a;
    } 
    else if (a.type == MESSAGE_STR) 
    {
        flag = true;
        a.session_type = "udp";
        return a;
    } 
    else if (a.type == MESSAGE_HTTP) 
    {
        flag = true;
        a.session_type = "http";
        return a;
    } 

    else {
        std::cout<<"undefined websocket message received"<<std::endl;// undefined.
    }
    return a;
}
bool CWSServer::send_msg(connection_hdl &send_hdl, const std::string& msg)
{
    unique_lock<mutex> lock(m_send_lock);
    if(m_connections.count(send_hdl)){
        std::cout<<"send msg: " << msg <<std::endl;
        m_server.send(send_hdl, msg, websocketpp::frame::opcode::text);
        return true;
    }else{
        std::cout<<"ws handle已删除, 客户端已经离线, 发送消息失败: " << msg <<std::endl;
        return false;
    }
}
bool CWSServer::bind_hdl_uuid(const connection_hdl &hdl, const std::string &uuid)
{
    lock_guard<mutex> guard(m_connection_lock);
    if(m_hdl_map.count(hdl)==0)
    {
        m_hdl_map[hdl] = uuid;
    }
    else
    {
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
